import fs from "node:fs/promises";
import path from "node:path";

import { TvQrcodeLogin } from "@renmu/bili-api";

function log(line) {
  try {
    process.stdout.write(String(line) + "\n");
  } catch {
    // ignore
  }
}

function parseArgs(argv) {
  const out = { output: "cookies.json" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--output" || a === "-o") {
      out.output = String(argv[i + 1] || "").trim() || out.output;
      i++;
      continue;
    }
    if (a === "--help" || a === "-h") {
      out.help = true;
      continue;
    }
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    log("用法: node scripts/biliapi/login.mjs [-o cookies.json]");
    process.exit(0);
  }

  const outPath = path.resolve(process.cwd(), args.output);
  const outDir = path.dirname(outPath);

  const tv = new TvQrcodeLogin();

  process.once("SIGINT", () => {
    log("\n收到 Ctrl+C，已中断登录。");
    try {
      tv.interrupt();
    } catch {
      // ignore
    }
    process.exit(130);
  });

  tv.on("scan", res => {
    if (res.code === 86039) log("等待扫码…");
    else if (res.code === 86090) log("已扫码，等待在手机确认…");
    else log(`扫码状态: code=${res.code} message=${res.message}`);
  });

  tv.on("completed", async res => {
    try {
      if (!res || !res.data) {
        throw new Error("登录完成，但返回 data 为空");
      }
      await fs.mkdir(outDir, { recursive: true });
      await fs.writeFile(outPath, JSON.stringify(res.data, null, 2), "utf-8");
      log(`登录成功，已保存 cookie 到: ${outPath}`);
    } catch (e) {
      log(`保存 cookie 失败: ${e?.message || String(e)}`);
      process.exit(1);
    }
  });

  tv.on("error", res => {
    log(`登录失败: code=${res?.code} message=${res?.message}`);
    process.exit(1);
  });

  const url = await tv.login();
  log("请用 Bilibili App 扫码登录（把下面 URL 转成二维码即可）：");
  log(url);
  log(`登录成功后会写入: ${outPath}`);
}

main().catch(e => {
  log(`异常: ${e?.message || String(e)}`);
  process.exit(1);
});

