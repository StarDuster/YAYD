import gradio as gr

from youdub.config import Settings
from youdub.core.pipeline import VideoPipeline
from youdub.core.steps import (
    download_from_url,
    generate_all_info_under_folder,
    generate_all_wavs_under_folder,
    separate_all_audio_under_folder,
    synthesize_all_video_under_folder,
    transcribe_all_audio_under_folder,
    translate_all_transcript_under_folder,
    upload_all_videos_under_folder,
)
from youdub.models import ModelCheckError, ModelManager

settings = Settings()
model_manager = ModelManager(settings)
pipeline = VideoPipeline(settings=settings, model_manager=model_manager)


def _safe_run(names, func, *args, **kwargs):
    try:
        model_manager.enforce_offline()
        if names:
            model_manager.ensure_ready(names)
        return func(*args, **kwargs)
    except ModelCheckError as exc:
        return str(exc)


def run_pipeline(
    root_folder,
    url,
    num_videos,
    resolution,
    demucs_model,
    device,
    shifts,
    whisper_model,
    whisper_batch_size,
    whisper_diarization,
    whisper_min_speakers,
    whisper_max_speakers,
    translation_target_language,
    tts_method,
    subtitles,
    speed_up,
    fps,
    target_resolution,
    max_workers,
    max_retries,
    auto_upload_video,
):
    try:
        return pipeline.run(
            root_folder=root_folder,
            url=url,
            num_videos=num_videos,
            resolution=resolution,
            demucs_model=demucs_model,
            device=device,
            shifts=shifts,
            whisper_model=whisper_model,
            whisper_batch_size=whisper_batch_size,
            whisper_diarization=whisper_diarization,
            whisper_min_speakers=whisper_min_speakers,
            whisper_max_speakers=whisper_max_speakers,
            translation_target_language=translation_target_language,
            tts_method=tts_method,
            subtitles=subtitles,
            speed_up=speed_up,
            fps=fps,
            target_resolution=target_resolution,
            max_workers=max_workers,
            max_retries=max_retries,
            auto_upload_video=auto_upload_video,
        )
    except ModelCheckError as exc:
        return str(exc)


def show_model_status():
    return model_manager.describe_status()


do_everything_interface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(label="Root Folder", value=str(settings.root_folder)),
        gr.Textbox(
            label="Video URL",
            placeholder="Video or Playlist or Channel URL",
            value="https://www.youtube.com/watch?v=4_SH2nfbQZ8",
        ),
        gr.Slider(minimum=1, maximum=500, step=1, label="Number of videos to download", value=20),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs Model",
            value=settings.demucs_model_name,
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Demucs Device", value=settings.demucs_device),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=settings.demucs_shifts),
        gr.Textbox(label="Whisper Model", value=str(settings.whisper_model_path)),
        gr.Slider(minimum=1, maximum=128, step=1, label="Whisper Batch Size", value=settings.whisper_batch_size),
        gr.Checkbox(label="Whisper Diarization", value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Min Speakers", value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Max Speakers", value=None),
        gr.Dropdown(
            ["简体中文", "繁体中文", "English", "Deutsch", "Français", "русский"],
            label="Translation Target Language",
            value=settings.translation_target_language,
        ),
        gr.Dropdown(
            ["bytedance", "qwen", "gemini"],
            label="TTS Method",
            value=settings.tts_method
        ),
        gr.Checkbox(label="Subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Speed Up", value=1.05),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Max Workers", value=1),
        gr.Slider(minimum=1, maximum=10, step=1, label="Max Retries", value=3),
        gr.Checkbox(label="Auto Upload Video", value=True),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

youtube_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(
            label="Video URL",
            placeholder="Video or Playlist or Channel URL",
            value="https://www.bilibili.com/list/1263732318",
        ),
        gr.Textbox(label="Output Folder", value=str(settings.root_folder)),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Number of videos to download", value=5),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

demucs_interface = gr.Interface(
    fn=lambda folder, model, device, progress, shifts: _safe_run(
        [model_manager._demucs_requirement().name],  # type: ignore[attr-defined]
        separate_all_audio_under_folder,
        folder,
        model_name=model,
        device=device,
        progress=progress,
        shifts=shifts,
        settings=settings,
        model_manager=model_manager,
    ),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Model",
            value=settings.demucs_model_name,
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value=settings.demucs_device),
        gr.Checkbox(label="Progress Bar in Console", value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=settings.demucs_shifts),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

whisper_inference = gr.Interface(
    fn=lambda folder, model, device, batch_size, diarization, min_speakers, max_speakers: _safe_run(
        [
            model_manager._whisper_requirement().name,  # type: ignore[attr-defined]
            *(  # include diarization requirement only when enabled
                [model_manager._whisper_diarization_requirement().name]  # type: ignore[attr-defined]
                if diarization
                else []
            ),
        ],
        transcribe_all_audio_under_folder,
        folder,
        model_name=model,
        device=device,
        batch_size=batch_size,
        diarization=diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        settings=settings,
        model_manager=model_manager,
    ),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Textbox(label="Model", value=str(settings.whisper_model_path)),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value=settings.demucs_device),
        gr.Slider(minimum=1, maximum=128, step=1, label="Batch Size", value=settings.whisper_batch_size),
        gr.Checkbox(label="Diarization", value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Min Speakers", value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Max Speakers", value=None),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

translation_interface = gr.Interface(
    fn=lambda folder, target_language: translate_all_transcript_under_folder(
        folder, target_language, settings=settings
    ),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Dropdown(
            ["简体中文", "繁体中文", "English", "Deutsch", "Français", "русский"],
            label="Target Language",
            value=settings.translation_target_language,
        ),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)


def _tts_wrapper(folder, tts_method):
    names = (
        [model_manager._bytedance_requirement().name]  # type: ignore[attr-defined]
        if tts_method == "bytedance"
        else [model_manager._gemini_tts_requirement().name]  # type: ignore[attr-defined]
        if tts_method == "gemini"
        else [
            model_manager._qwen_tts_runtime_requirement().name,  # type: ignore[attr-defined]
            model_manager._qwen_tts_weights_requirement().name,  # type: ignore[attr-defined]
        ]
        if tts_method == "qwen"
        else []
    )
    return _safe_run(
        names,
        generate_all_wavs_under_folder,
        folder,
        tts_method=tts_method,
    )


tts_interface = gr.Interface(
    fn=_tts_wrapper,
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Dropdown(
            ["bytedance", "qwen", "gemini"],
            label="TTS Method",
            value=settings.tts_method
        ),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

syntehsize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Checkbox(label="Subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Speed Up", value=1.05),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

genearte_info_interface = gr.Interface(
    fn=generate_all_info_under_folder,
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

upload_bilibili_interface = gr.Interface(
    fn=upload_all_videos_under_folder,
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
    ],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
)

model_status_interface = gr.Interface(
    fn=show_model_status,
    inputs=[],
    outputs=gr.Textbox(label="Output", lines=5, max_lines=100, autoscroll=True),
    description="当前需要的 ASR/TTS 模型状态（仅本地，不会自动下载）。",
)

app = gr.TabbedInterface(
    interface_list=[
        model_status_interface,
        do_everything_interface,
        youtube_interface,
        demucs_interface,
        whisper_inference,
        translation_interface,
        tts_interface,
        syntehsize_video_interface,
        genearte_info_interface,
        upload_bilibili_interface,
    ],
    tab_names=['模型检查', '全自动', '下载视频', '人声分离', '语音识别', '字幕翻译', '语音合成', '视频合成', '生成信息', '上传B站'],
    title='YouDub',
)


def main():
    app.launch()


if __name__ == "__main__":
    main()
