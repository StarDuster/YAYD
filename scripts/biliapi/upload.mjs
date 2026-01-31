import fs from "node:fs/promises";
import path from "node:path";
import http from "node:http";
import https from "node:https";

import axios from "axios";
import ProxyAgent from "proxy-agent";

import { Auth, Client, WebVideoUploader, BiliResponseError } from "@renmu/bili-api";

const RESULT_PREFIX = "__YOUDUB_RESULT__";

function log(line) {
  try {
    process.stderr.write(String(line) + "\n");
  } catch {
    // ignore
  }
}

function emitResult(obj) {
  process.stdout.write(`${RESULT_PREFIX}${JSON.stringify(obj)}\n`);
}

function isObject(x) {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

function pickFlatCookieFields(obj) {
  const out = {};
  if (!isObject(obj)) return out;
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v === "string" || typeof v === "number") out[k] = v;
  }
  return out;
}

function setupProxy(proxyUrl) {
  const raw = String(proxyUrl || "").trim();
  if (!raw) return;
  const agent = new ProxyAgent(raw);

  // Make native http/https use proxy by default when no agent is provided.
  // NOTE: This affects *only this Node process* (the upload script).
  http.globalAgent = agent;
  https.globalAgent = agent;

  // Patch axios.create so every BaseRequest inside biliAPI uses proxy agent.
  const oldCreate = axios.create.bind(axios);
  axios.create = (config = {}) =>
    oldCreate({
      ...config,
      // Let agent handle proxy, disable axios proxy option to avoid conflicts.
      proxy: false,
      httpAgent: agent,
      httpsAgent: agent,
    });

  log(`[biliapi] 使用代理: ${raw}`);
}

async function loadAuthFromCookieFile(cookieFile, accessTokenOverride) {
  const auth = new Auth();
  const raw = await fs.readFile(cookieFile, "utf-8");

  let obj;
  try {
    obj = JSON.parse(raw);
  } catch (e) {
    throw new Error(`cookie 文件不是合法 JSON: ${cookieFile}`);
  }

  // TvQrcodeLogin/WebQrcodeLogin 格式（biliAPI 原生）
  if (isObject(obj) && isObject(obj.cookie_info) && Array.isArray(obj.cookie_info.cookies) && isObject(obj.token_info)) {
    await auth.loadCookieFile(cookieFile);
    return auth;
  }

  // 兼容平铺 cookie 格式（抓包导出的对象）
  const flat = pickFlatCookieFields(obj);
  const sess = flat.SESSDATA || flat.sessdata;
  const jct = flat.bili_jct || flat.csrf || flat.biliJct;
  if (!sess || !jct) {
    throw new Error(
      "cookie JSON 缺少 SESSDATA / bili_jct。请用 `node scripts/biliapi/login.mjs` 重新登录生成 cookies.json，或提供包含这两个字段的 cookie 文件。"
    );
  }

  const uidRaw = flat.DedeUserID || flat.uid || flat.mid || 0;
  const uid = Number(uidRaw) || 0;
  const accessToken = accessTokenOverride || flat.access_token || flat.accessToken;

  // Auth.setAuth 会把 cookie 对象序列化为 "k=v; ..." 形式。
  const cookie = {
    ...flat,
    SESSDATA: String(sess),
    bili_jct: String(jct),
  };
  auth.setAuth(cookie, uid, typeof accessToken === "string" ? accessToken : undefined);
  return auth;
}

function normalizeLine(line) {
  const s = String(line || "").trim().toLowerCase();
  if (!s) return "auto";
  // 常见别名映射到 biliAPI 支持的值：auto/bda2/qn/qnhk/bldsa/...
  const aliases = {
    ali: "auto",
    alia: "auto",
    bda: "bda2",
    tx: "auto",
    txa: "auto",
  };
  return aliases[s] || s;
}

function normalizeSubmitOrder(order) {
  const raw = Array.isArray(order) ? order : [];
  const out = [];
  for (const item of raw) {
    const s = String(item || "").trim().toLowerCase();
    if (!s) continue;
    if (s === "app") out.push("web");
    else if (s === "b-cut-android" || s === "bcut" || s === "bcut-android") out.push("b-cut");
    else out.push(s);
  }
  // default: web first, then b-cut fallback
  if (!out.length) out.push("web", "b-cut");
  // de-dupe
  return [...new Set(out)];
}

function describeError(err) {
  if (err instanceof BiliResponseError) {
    const meta = err.meta || {};
    return {
      name: err.name,
      message: err.message,
      statusCode: meta.statusCode,
      code: meta.code,
      path: meta.path,
      method: meta.method,
    };
  }
  if (err instanceof Error) {
    return { name: err.name, message: err.message, stack: err.stack };
  }
  return { message: String(err) };
}

async function main() {
  const stdin = await new Promise(resolve => {
    let data = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", chunk => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", () => resolve(data));
  });

  if (!stdin || !String(stdin).trim()) {
    emitResult({ ok: false, error: "stdin 为空：请通过 stdin 传入 JSON 参数" });
    process.exit(2);
  }

  let payload;
  try {
    payload = JSON.parse(stdin);
  } catch (e) {
    emitResult({ ok: false, error: "stdin 不是合法 JSON" });
    process.exit(2);
  }

  try {
    const cookieFile = String(payload.cookieFile || "").trim();
    if (!cookieFile) throw new Error("缺少 cookieFile");

    const videoPaths = Array.isArray(payload.videoPaths) ? payload.videoPaths.map(String) : [];
    if (!videoPaths.length) throw new Error("videoPaths 不能为空");

    const options = payload.options;
    if (!isObject(options)) throw new Error("options 必须是对象");
    if (!options.title) throw new Error("options.title 必填");
    if (!options.tid) throw new Error("options.tid 必填");
    if (!options.tag) throw new Error("options.tag 必填");

    setupProxy(payload.proxy);

    const auth = await loadAuthFromCookieFile(cookieFile, payload.accessToken);
    const client = new Client(auth);

    const upload = isObject(payload.upload) ? payload.upload : {};
    const line = normalizeLine(upload.line);
    const retryTimes = Number.isFinite(upload.retryTimes) ? Number(upload.retryTimes) : 3;
    const retryDelay = Number.isFinite(upload.retryDelay) ? Number(upload.retryDelay) : 3000;
    const concurrency = Number.isFinite(upload.concurrency) ? Number(upload.concurrency) : 3;
    const bcutPreUpload =
      upload.bcutPreUpload === false ? false : true; // default true: avoid 406 风控

    const parts = videoPaths.map((p, idx) => {
      const t = Array.isArray(payload.partTitles) ? payload.partTitles[idx] : undefined;
      return {
        path: p,
        title: typeof t === "string" && t.trim() ? t.trim() : path.parse(p).name,
      };
    });

    let currentUploader = null;
    let cancelled = false;
    const onSignal = sig => {
      cancelled = true;
      log(`[biliapi] 收到信号 ${sig}，尝试取消上传…`);
      try {
        if (currentUploader) currentUploader.cancel();
      } catch {
        // ignore
      }
    };
    process.once("SIGINT", () => onSignal("SIGINT"));
    process.once("SIGTERM", () => onSignal("SIGTERM"));

    const uploaded = [];
    for (let i = 0; i < parts.length; i++) {
      if (cancelled) throw new Error("已取消");
      const part = parts[i];
      log(`[biliapi] 上传分 P ${i + 1}/${parts.length}: ${part.title}`);

      const uploader = new WebVideoUploader(part, auth, {
        concurrency,
        retryTimes,
        retryDelay,
        line,
        // 使用必剪预上传接口规避 406 风控
        bcutPreUpload,
      });
      currentUploader = uploader;

      let lastPct = -1;
      uploader.emitter.on("progress", p => {
        if (!p || p.event !== "uploading") return;
        const pct = Math.floor((p.progress || 0) * 100);
        if (pct !== lastPct && (pct % 2 === 0 || pct === 100)) {
          lastPct = pct;
          log(`[biliapi] 上传进度: ${pct}%`);
        }
      });

      const res = await uploader.upload();
      uploaded.push(res);
      log(`[biliapi] 分 P 上传完成: cid=${res.cid} filename=${res.filename}`);
    }

    const submitOrder = normalizeSubmitOrder(payload.submitOrder);
    let submitLastError = null;

    for (const submit of submitOrder) {
      if (cancelled) throw new Error("已取消");
      log(`[biliapi] 开始投稿（submit=${submit}）…`);
      try {
        let out;
        if (submit === "web") {
          out = await client.platform.addMediaWebApi(uploaded, options);
        } else if (submit === "web-v3" || submit === "webv3" || submit === "v3") {
          out = await client.platform.addMediaWebApiV3(uploaded, options);
        } else if (submit === "b-cut" || submit === "bcut") {
          out = await client.platform.addMediaBCutApi(uploaded, options);
        } else if (submit === "client") {
          out = await client.platform.addMediaClientApi(uploaded, options);
        } else {
          throw new Error(`未知 submit 类型: ${submit}`);
        }

        emitResult({
          ok: true,
          submit,
          aid: out?.aid,
          bvid: out?.bvid,
        });
        return;
      } catch (e) {
        const info = describeError(e);
        submitLastError = info;
        log(`[biliapi] 投稿失败（submit=${submit}）：${info.message || String(e)}`);
        // 继续尝试下一个 submit
      }
    }

    throw new Error(
      `所有投稿接口均失败（${submitOrder.join(", ")}）。最后一次错误：${JSON.stringify(submitLastError)}`
    );
  } catch (e) {
    emitResult({ ok: false, error: describeError(e) });
    process.exit(1);
  }
}

main();

