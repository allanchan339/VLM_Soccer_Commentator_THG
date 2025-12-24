#!/usr/bin/env python3
"""
Merged Gradio demo for soccer video analysis and inpainting/TTS.
This single file combines the functionality of app_all.py and gradio_demo.py.
"""

# =================== IMPORTS & CONFIGURATION ===================
import os
from typing import Optional, Tuple, List
from dotenv import load_dotenv
import gradio as gr
import sys
import asyncio
import argparse
import time

# --- Imports from SoComm for Inpainting/TTS ---
from SoComm.models import load_all_models
from SoComm.realtime_inpainting import create_realtime_avatar, realtime_inference
from musetalk.utils.blending import get_image_blending
from SoComm.tts import tts_generate
from SoComm.utils import fast_check_ffmpeg, check_video, ensure_directory_exists, get_video_length, load_gallery_videos, tts_to_audio
from SoComm.video_analyzer import VideoAnalyzer

# --- GPT-SoVITS imports ---
GPT_SOVITS_AVAILABLE = False
gpt_sovits_pipeline = None

# Try to import GPT-SoVITS components
# Note: This is optional and won't break the main application if it fails

# Add GPT-SoVITS to Python path
gpt_sovits_base = os.path.join(os.path.dirname(__file__), "GPT-SoVITS")
gpt_sovits_path = os.path.join(gpt_sovits_base, "GPT_SoVITS")

if os.path.exists(gpt_sovits_path):
    print(f"Found GPT-SoVITS at: {gpt_sovits_path}")
    
    # Add paths to sys.path (following the original approach)
    if gpt_sovits_base not in sys.path:
        sys.path.append(gpt_sovits_base)
    if gpt_sovits_path not in sys.path:
        sys.path.append(gpt_sovits_path)
    if os.path.join(gpt_sovits_path, "eres2net") not in sys.path:
        sys.path.append(os.path.join(gpt_sovits_path, "eres2net"))
    
    print(f"Added GPT-SoVITS paths to Python path")
    
    # Import GPT-SoVITS components
    import sys
    sys.path.insert(0, os.path.join(gpt_sovits_path, "eres2net"))
    from ERes2NetV2 import ERes2NetV2
    print("✅ ERes2NetV2 import successful")
    
    from TTS_infer_pack.text_segmentation_method import get_method
    from TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config
    from tools.assets import css, js, top_html
    from tools.i18n.i18n import I18nAuto, scan_language_list
    from config import change_choices, get_weights_names, name2gpt_path, name2sovits_path
    from process_ckpt import get_sovits_version_from_path_fast
    
    print("✅ GPT-SoVITS imports successful")
    GPT_SOVITS_AVAILABLE = True
    
else:
    print(f"GPT-SoVITS path not found: {gpt_sovits_path}")
    GPT_SOVITS_AVAILABLE = False

if not GPT_SOVITS_AVAILABLE:
    print("ℹ️ GPT-SoVITS features will be disabled. SoComm TTS will continue to work normally.")
else:
    print("🎉 GPT-SoVITS is fully loaded and ready! Voice cloning features are now available.")

# --- Load environment variables for Video Analysis ---
load_dotenv(override=True)

# Assert that MODELSCOPE_SDK_TOKEN is properly loaded
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")
assert MODELSCOPE_SDK_TOKEN, "MODELSCOPE_SDK_TOKEN not found in environment variables. Please check your .env file and ensure python-dotenv is installed."

# =================== MODEL & APP CONFIGURATION ===================

def parse_args():
    parser = argparse.ArgumentParser(description="Soccer Video Analysis and Inpainting/TTS Demo")
    # General paths and server
    parser.add_argument('--ffmpeg_path', type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help='Path to ffmpeg executable')
    parser.add_argument('--use_float16', action='store_true', default=True, help='Use float16 for models')
    parser.add_argument('--gallery_dir', type=str, default="video_gallery", help='Directory for video gallery')
    parser.add_argument('--processed_videos_dir', type=str, default="processed_videos", help='Directory for processed videos')
    parser.add_argument('--video_extensions', type=str, nargs='+', default=['.mp4', '.avi', '.mov', '.mkv'], help='Allowed video extensions')
    parser.add_argument('--default_server_name', type=str, default="0.0.0.0", help='Gradio server name')
    parser.add_argument('--default_server_port', type=int, default=7869, help='Gradio server port')
    parser.add_argument('--default_share', action='store_true', default=True, help='Gradio share option')
    # Model loading
    parser.add_argument('--unet_model_path', type=str, default="./models/musetalkV15/unet.pth", help='Path to UNet model weights')
    parser.add_argument('--vae_type', type=str, default="sd-vae", help='Type of VAE model')
    parser.add_argument('--unet_config', type=str, default="./models/musetalkV15/musetalk.json", help='Path to UNet config')
    parser.add_argument('--whisper_dir', type=str, default="./models/whisper", help='Directory for Whisper model')
    # Video analysis
    parser.add_argument('--qwen_model', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct", help='Qwen model name')
    parser.add_argument('--modelscope_base_url', type=str, default="https://api-inference.modelscope.cn/v1", help='Modelscope API base URL')
    parser.add_argument('--target_words_per_second', type=int, default=4, help='Target words per second for commentary')
    # Inpainting/Generation parameters
    parser.add_argument('--inpainting_result_dir', type=str, default='./results/output', help='Directory for inpainting results')
    parser.add_argument('--debug_result_dir', type=str, default='./results/debug', help='Directory for debug inpainting results')
    parser.add_argument('--inpainting_fps', type=int, default=25, help='FPS for inpainting output')
    parser.add_argument('--inpainting_batch_size', type=int, default=20, help='Batch size for inpainting')
    parser.add_argument('--inpainting_output_vid_name', type=str, default='', help='Output video name for inpainting')
    parser.add_argument('--inpainting_use_saved_coord', action='store_true', default=False, help='Use saved coordinates for inpainting')
    parser.add_argument('--inpainting_audio_padding_left', type=int, default=2, help='Audio left padding for inpainting')
    parser.add_argument('--inpainting_audio_padding_right', type=int, default=2, help='Audio right padding for inpainting')
    parser.add_argument('--inpainting_version', type=str, default='v15', help='Inpainting model version')
    # Video processing/check_video
    parser.add_argument('--results_dir', type=str, default='./results', help='General results directory')
    parser.add_argument('--output_dir', type=str, default='./results/output', help='General output directory')
    parser.add_argument('--input_dir', type=str, default='./results/input', help='General input directory')
    
    # GPT-SoVITS specific arguments
    parser.add_argument('--gpt_model_path', type=str, default=None, help='Path to GPT model weights')
    parser.add_argument('--sovits_model_path', type=str, default=None, help='Path to SoVITS model weights')
    parser.add_argument('--cnhubert_base_path', type=str, default=None, help='Path to CNHubert base model')
    parser.add_argument('--bert_path', type=str, default=None, help='Path to BERT model')
    parser.add_argument('--gpt_sovits_version', type=str, default='v2', help='GPT-SoVITS version (v1/v2)')
    parser.add_argument('--use_gpt_sovits', action='store_true', default=True, help='Enable GPT-SoVITS TTS')
    return parser.parse_args()

# --- Configuration for Video Analysis ---
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
GALLERY_DIR = "video_gallery"
PROCESSED_VIDEOS_DIR = "processed_videos"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
TARGET_WORDS_PER_SECOND = 4

# --- Gradio Server Config ---
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 7869
DEFAULT_SHARE = True

# --- GPT-SoVITS Configuration ---
# Language dictionaries for different versions
dict_language_v1 = {
    "中文": "all_zh",  # 全部按中文识别
    "英文": "en",  # 全部按英文识别
    "日文": "all_ja",  # 全部按日文识别
    "中英混合": "zh",  # 按中英混合识别
    "日英混合": "ja",  # 按日英混合识别
    "多语种混合": "auto",  # 多语种启动切分识别语种
}

dict_language_v2 = {
    "中文": "all_zh",  # 全部按中文识别
    "英文": "en",  # 全部按英文识别
    "日文": "all_ja",  # 全部按日文识别
    "粤语": "all_yue",  # 全部按粤语识别
    "韩文": "all_ko",  # 全部按韩文识别
    "中英混合": "zh",  # 按中英混合识别
    "日英混合": "ja",  # 按日英混合识别
    "粤英混合": "yue",  # 按粤英混合识别
    "韩英混合": "ko",  # 按韩英混合识别
    "多语种混合": "auto",  # 多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",  # 多语种启动切分识别语种
}

# Text segmentation methods
cut_method = {
    "不切": "cut0",
    "凑四句一切": "cut1",
    "凑50字一切": "cut2",
    "按中文句号。切": "cut3",
    "按英文句号.切": "cut4",
    "按标点符号切": "cut5",
}

# =================== MODEL LOADING & SETUP ===================

args = parse_args()

# --- Load Inpainting/TTS models ---
print("Loading SoComm models for TTS & THG...")
(device, vae, unet, pe, weight_dtype, audio_processor, whisper, timesteps) = load_all_models(
    use_float16=args.use_float16,
    unet_model_path=args.unet_model_path,
    vae_type=args.vae_type,
    unet_config=args.unet_config,
    whisper_dir=args.whisper_dir
)
print("SoComm models loaded successfully.")

# --- GPT-SoVITS Model Setup ---
# Language dictionaries for different versions
dict_language = dict_language_v2  # Default to v2 for better language support

def resolve_gpt_sovits_paths():
    """Resolve GPT-SoVITS model paths from different web UI configurations"""
    if not GPT_SOVITS_AVAILABLE:
        return None
        
    # Try multiple possible path configurations based on file analysis
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "GPT-SoVITS", "GPT_SoVITS"),  # Same directory as web_ui_all.py
        "./GPT-SoVITS/GPT_SoVITS/",  # Relative to current working directory
        "../GPT-SoVITS/GPT_SoVITS/", # One level up
        os.path.join(os.getcwd(), "GPT-SoVITS", "GPT_SoVITS"), # From current working directory
    ]
    
    for base_path in possible_paths:
        # Check if this is the GPT_SoVITS subdirectory (contains configs/tts_infer.yaml)
        if os.path.exists(os.path.join(base_path, "configs/tts_infer.yaml")):
            print(f"Found GPT-SoVITS config at: {base_path}")
            return base_path
    
    return None

def load_gpt_sovits_models():
    """Load GPT-SoVITS models only when needed (lazy loading)"""
    global gpt_sovits_pipeline, dict_language
    
    if gpt_sovits_pipeline is None and GPT_SOVITS_AVAILABLE:
        try:
            base_path = resolve_gpt_sovits_paths()
            if not base_path:
                raise FileNotFoundError("GPT-SoVITS directory not found in expected locations")
                
            config_path = os.path.join(base_path, "configs/tts_infer.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            print(f"Loading GPT-SoVITS from: {base_path}")
            print(f"Using config: {config_path}")
            
            # Change to the parent directory (GPT-SoVITS) not the subdirectory (GPT_SoVITS)
            # This is because the config expects to be run from GPT-SoVITS/, not GPT-SoVITS/GPT_SoVITS/
            parent_dir = os.path.dirname(base_path)  # This should be GPT-SoVITS/
            original_cwd = os.getcwd()
            os.chdir(parent_dir)
            
            try:
                # Use config relative to parent directory
                config_relative_path = os.path.join(os.path.basename(base_path), "configs/tts_infer.yaml")
                gpt_sovits_config = TTS_Config(config_relative_path)
                gpt_sovits_config.device = device  # Use shared device from SoComm
                gpt_sovits_config.is_half = args.use_float16
                gpt_sovits_config.update_version(args.gpt_sovits_version)
                
                # The config file already has the correct relative paths
                # They expect to be run from GPT-SoVITS/ directory
                
                gpt_sovits_pipeline = TTS(gpt_sovits_config)
                
                # Update language dictionary based on version
                if args.gpt_sovits_version == "v1":
                    dict_language = dict_language_v1
                else:
                    dict_language = dict_language_v2
                    
                print("✅ GPT-SoVITS models loaded successfully with default models")
                
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"❌ Failed to load GPT-SoVITS models: {e}")
            gpt_sovits_pipeline = None

# --- Check ffmpeg and add to PATH ---
if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find or configure ffmpeg. Video generation may fail.")

# --- Solve asynchronous IO issues on Windows ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =================== UTILITY FUNCTIONS ===================

def load_avatar_gallery(avatars_dir="results/v15/avatars"):
    """Load avatar gallery from the avatars directory."""
    gallery_items = []
    
    if not os.path.exists(avatars_dir):
        return gallery_items
    
    for avatar_name in os.listdir(avatars_dir):
        avatar_path = os.path.join(avatars_dir, avatar_name)
        if os.path.isdir(avatar_path):
            # Check if mask_coords.pkl exists (indicates avatar is ready for inference)
            mask_coords_path = os.path.join(avatar_path, "mask_coords.pkl")
            if not os.path.exists(mask_coords_path):
                continue  # Skip this avatar if mask_coords.pkl doesn't exist
            
            # Look for the first image in full_imgs directory
            full_imgs_path = os.path.join(avatar_path, "full_imgs")
            if os.path.exists(full_imgs_path):
                # Try to find 00000001.png first, then any other image
                target_image = os.path.join(full_imgs_path, "00000001.png")
                if not os.path.exists(target_image):
                    # Find any image file
                    for img_file in os.listdir(full_imgs_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            target_image = os.path.join(full_imgs_path, img_file)
                            break
                
                if os.path.exists(target_image):
                    gallery_items.append((target_image, avatar_name))
    
    return gallery_items

def unified_tts_generation(
    text, voice_type, 
    # SoComm parameters
    tts_voice=None,
    # GPT-SoVITS parameters (using exact defaults from inference_webui_fast.py)
    ref_audio=None, aux_refs=None, prompt_text=None, prompt_lang=None,
    target_lang=None, top_k=5, top_p=1, temperature=1, speed_factor=1.0, 
    batch_size=20, sample_steps=32, fragment_interval=0.3, how_to_cut="凑四句一切",
    super_sampling=False, parallel_infer=True, split_bucket=True, seed=-1, 
    keep_random=True, repetition_penalty=1.35
):
    """Unified TTS function that routes to appropriate system"""
    
    # Create consistent audio output directory
    audio_output_dir = "temp_audio"
    os.makedirs(audio_output_dir, exist_ok=True)
    
    if voice_type == "socomm":
        # Use existing SoComm TTS
        return tts_to_audio(text, tts_voice)
    
    elif voice_type == "gpt_sovits":
        # Use GPT-SoVITS TTS
        if not ref_audio:
            gr.Warning("Reference audio required for GPT-SoVITS")
            return None
            
        # Ensure models are loaded
        if gpt_sovits_pipeline is None:
            load_gpt_sovits_models()
            if gpt_sovits_pipeline is None:
                gr.Error("Failed to load GPT-SoVITS models")
                return None
            
        inputs = {
            "text": text,
            "text_lang": dict_language[target_lang],
            "ref_audio_path": ref_audio,
            "aux_ref_audio_paths": [item.name for item in aux_refs] if aux_refs else [],
            "prompt_text": prompt_text or "",
            "prompt_lang": dict_language[prompt_lang],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "batch_size": batch_size,
            "sample_steps": sample_steps,
            "fragment_interval": fragment_interval,
            "text_split_method": cut_method[how_to_cut],
            "super_sampling": super_sampling,
            "parallel_infer": parallel_infer,
            "split_bucket": split_bucket,
            "seed": seed,
            "keep_random": keep_random,
            "repetition_penalty": repetition_penalty,
            "ref_text_free": False,
            "return_fragment": False
        }
        
        try:
            result = gpt_sovits_pipeline.run(inputs)
            # Process result and return audio path
            # GPT-SoVITS yields (sr, audio) tuples directly
            for item in result:
                if isinstance(item, tuple) and len(item) == 2:
                    sampling_rate, audio_data = item
                    # Save the audio data to a consistent audio output directory
                    import soundfile as sf
                    import time
                    
                    # Create GPT-SoVITS subdirectory
                    gpt_sovits_dir = os.path.join(audio_output_dir, "gpt_sovits")
                    os.makedirs(gpt_sovits_dir, exist_ok=True)
                    
                    # Generate unique filename based on timestamp
                    timestamp = int(time.time())
                    audio_filename = f"gpt_sovits_{timestamp}.wav"
                    audio_path = os.path.join(gpt_sovits_dir, audio_filename)
                    
                    # Save the audio file
                    sf.write(audio_path, audio_data, sampling_rate)
                    
                    print(f"✅ GPT-SoVITS audio saved to: {audio_path}")
                    return audio_path
                elif isinstance(item, str) and os.path.exists(item):
                    return item
                elif hasattr(item, 'name') and os.path.exists(item.name):
                    return item.name
            
            # If no audio found, return None
            print("⚠️ No audio data found in GPT-SoVITS output")
            return None
        except Exception as e:
            gr.Error(f"GPT-SoVITS error: {str(e)}")
            return None

# =================== SERVICES (for Video Analysis) ===================

# =================== MERGED GRADIO UI ===================

def merged_interface(args):
    """Defines the integrated Gradio UI for the soccer demo."""
    css = """#output_vid {max-width: 1024px; max-height: 576px}"""
    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# Soccer Commentary & Talking Head Video Generation Demo")
        with gr.Row(equal_height=False):
            # --- LEFT COLUMN: All user actions/inputs ---
            with gr.Column(scale=1, elem_id="left-col"):
                gr.Markdown("### 1. Select or Upload Soccer Clip")
                
                # Navigation tabs for soccer clip input/gallery
                with gr.Tabs(elem_id="video_tabs") as video_tabs:
                    with gr.TabItem("1. Upload Video"):
                        video_input = gr.Video(label="Upload Soccer Clip", interactive=True)
                    
                    with gr.TabItem("2. Video Gallery"):
                        gr.Markdown("**Select from existing soccer clips:**")
                        gallery = gr.Gallery(
                            label="Video Gallery",
                            show_label=True,
                            elem_id="video_gallery",
                            value=load_gallery_videos(args.gallery_dir, args.video_extensions),
                            columns=4, rows=2, object_fit="contain", allow_preview=False
                        )
                        refresh_gallery_btn = gr.Button("🔄 Refresh Video Gallery", size="sm")
                
                # System prompt textbox for video analysis
                default_system_prompt = "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game.  Describe this game scene, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. You should also have an engaging tone. SKIP all non commentary content, 用廣東話回答, 不要使用英文, MAKE SURE YOU ARE SPOTTING CORRECT ACTIONS BEFORE ANSWERING, ALSO THE SPEECH MUST BE ENERGETIC"
                system_prompt_box = gr.Textbox(label="System Prompt for Soccer Commentary AI", value=default_system_prompt, lines=3)
                analyze_btn = gr.Button("Analyze Video", variant="primary")
                
                commentary_tts_box = gr.Textbox(label="Generated Commentary / Text for TTS", placeholder="Commentary will appear here, or type your own text for TTS...", lines=4, interactive=True)
                
                # --- TTS MODULE ---
                gr.Markdown("### 2. Text-to-Speech (TTS)")
                
                # TTS Selection Tabs
                with gr.Tabs():
                    with gr.TabItem("1. SoComm TTS (Basic)"):
                        # Existing SoComm TTS interface - unchanged
                        tts_voice = gr.Dropdown(
                            label="TTS Voice",
                            choices=[
                                "en-US-AriaNeural",
                                "en-US-GuyNeural",
                                "zh-CN-XiaoxiaoNeural",
                                "zh-CN-YunxiNeural",
                                "zh-HK-HiuGaaiNeural",
                                "zh-HK-HiuMaanNeural",
                                "zh-HK-WanLungNeural"
                            ],
                            value="zh-HK-WanLungNeural"
                        )
                        tts_btn = gr.Button("Synthesize Audio (SoComm)")
                    
                    if GPT_SOVITS_AVAILABLE:
                        with gr.TabItem("2. GPT-SoVITS TTS (Voice Cloning)"):
                            # New GPT-SoVITS interface
                            gr.Markdown("**Clone voice from reference audio for natural TTS**")
                            
                            # Reference Audio Upload
                            with gr.Row():
                                gpt_ref_audio = gr.Audio(
                                    label="Reference Audio (3-10 seconds)", 
                                    type="filepath",
                                    interactive=True
                                )
                                gpt_aux_refs = gr.File(
                                    label="Auxiliary Reference Audio (Optional)", 
                                    file_count="multiple",
                                    visible=False  # Hidden by default
                                )
                            
                            # Reference Text
                            gpt_prompt_text = gr.Textbox(
                                label="Reference Audio Text (Optional)", 
                                value="", 
                                lines=2,
                                
                            )
                            
                            # Language Selection (Yue as default)
                            with gr.Row():
                                gpt_prompt_lang = gr.Dropdown(
                                    label="Reference Language", 
                                    choices=list(dict_language.keys()), 
                                    value="粤语"
                                )
                                gpt_target_lang = gr.Dropdown(
                                    label="Target Language", 
                                    choices=list(dict_language.keys()), 
                                    value="粤语"
                                )
                            
                            # Advanced Parameters (Hidden by default, using exact defaults from inference_webui_fast.py)
                            with gr.Accordion("Advanced Settings (Optional)", open=False):
                                with gr.Row():
                                    gpt_batch_size = gr.Slider(minimum=1, maximum=200, step=1, label="Batch Size", value=20)
                                    gpt_sample_steps = gr.Radio(label="Sample Steps (V3/4 only)", value=32, choices=[4, 8, 16, 32, 64, 128])
                                
                                with gr.Row():
                                    gpt_fragment_interval = gr.Slider(minimum=0.01, maximum=1, step=0.01, label="Fragment Interval (s)", value=0.3)
                                    gpt_speed_factor = gr.Slider(minimum=0.6, maximum=1.65, step=0.05, label="Speed Factor", value=1.25)
                                
                                with gr.Row():
                                    gpt_top_k = gr.Slider(minimum=1, maximum=100, step=1, label="Top-K", value=5)
                                    gpt_top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label="Top-P", value=1)
                                
                                with gr.Row():
                                    gpt_temperature = gr.Slider(minimum=0, maximum=1, step=0.05, label="Temperature", value=1)
                                    gpt_repetition_penalty = gr.Slider(minimum=0, maximum=2, step=0.05, label="Repetition Penalty", value=1.35)
                                
                                with gr.Row():
                                    gpt_how_to_cut = gr.Dropdown(
                                        label="Text Segmentation Method",
                                        choices=list(cut_method.keys()),
                                        value="凑四句一切"
                                    )
                                    gpt_super_sampling = gr.Checkbox(label="Audio Super Sampling (V3 only)", value=False)
                                
                                with gr.Row():
                                    gpt_parallel_infer = gr.Checkbox(label="Parallel Inference", value=True)
                                    gpt_split_bucket = gr.Checkbox(label="Data Bucketing", value=True)
                                
                                with gr.Row():
                                    gpt_seed = gr.Number(label="Random Seed", value=-1)
                                    gpt_keep_random = gr.Checkbox(label="Keep Random", value=True)
                            
                            gpt_tts_btn = gr.Button("Generate Speech (GPT-SoVITS)", variant="primary")
                            
                            # Note: Using default models from inference_webui_fast.py
                            # No model management UI needed - models are loaded automatically
                
                # Hidden trigger for client-side tab switching
                tab_switcher = gr.Number(visible=False, value=0)

                # Progress tracking component
                progress_tracker = gr.Progress()
                
                # --- AUDIO BLOCK: TTS Output & Driving Audio for Video Generation ---
                gr.Markdown("### 3. Talking Head Video Generation")
                driving_audio = gr.Audio(label="Driving Audio (TTS output or upload your own)", type="filepath", interactive=True)
                
                gr.Markdown("### 3.1. Avatar Creation (MuseV)")
                
                # Navigation tabs for avatar creation/gallery
                with gr.Tabs():
                    with gr.TabItem("1. Avatar Creation"):
                        avatar_id_input = gr.Textbox(label="Avatar ID", placeholder="Enter unique avatar name", value="my_avatar")
                        avatar_video_input = gr.Video(label="Avatar Reference Video (Upload face video for avatar creation)", sources=['upload'])
                        
                        
                        create_avatar_btn = gr.Button("Create Avatar (Which consumes time)", variant="primary")
                    
                    with gr.TabItem("2. Avatar Gallery"):
                        gr.Markdown("**Select from existing avatars:**")
                        avatar_gallery = gr.Gallery(
                            label="Avatar Gallery",
                            show_label=True,
                            elem_id="avatar_gallery",
                            value=load_avatar_gallery(),
                            columns=3, rows=2, object_fit="contain", allow_preview=False
                        )
                        refresh_avatar_gallery_btn = gr.Button("🔄 Refresh Avatar Gallery", size="sm")
                
                    realtime_generate_btn = gr.Button("Generate Talking Head", variant="primary")

                # Inpainting parameter controls (hidden by default, can be shown if needed)
                # Move these to the bottom of the left column in a dropdown
                # bbox_shift = gr.Number(label="BBox Shift (px)", value=0, visible=False)
                # extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1, visible=False)
                # parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw", visible=False)
                # left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5, visible=False)
                # right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5, visible=False)
                # Place at the bottom of the left column
                with gr.Accordion("Advanced Inpainting Parameters", open=False):
                    bbox_shift = gr.Number(label="BBox Shift (px)", value=0)
                    extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
                    parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
                    left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
                    right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)

            # --- RIGHT COLUMN: All outputs/results ---
            with gr.Column(scale=1, elem_id="right-col"):
                # Real-time avatar output
                gr.Markdown("### Real-Time Avatar Output")
                realtime_output_video = gr.Video(label="Real-Time Generated Video")
                
                # Parameter Information as dropdown/accordion
                with gr.Accordion("Running Information", open=True):
                    debug_output_info = gr.Textbox(label="Console Output", lines=4)
                


        # =================== EVENT HANDLERS ===================
        video_analyzer = VideoAnalyzer(
            qwen_model=args.qwen_model,
            modelscope_base_url=args.modelscope_base_url,
            target_words_per_second=args.target_words_per_second,
            modelscope_token=MODELSCOPE_SDK_TOKEN
        )
        
        # Helper functions for real-time inpainting
        
        def process_analysis_video(video_path, system_prompt, progress=gr.Progress(track_tqdm=True)):
            if not video_path:
                gr.Info("❌ No video provided. Please upload or select a video first.")
                return "No video provided. Please upload or select a video first."
            
            try:
                gr.Info("🔄 Starting video analysis... This may take a few moments.")
                commentary = video_analyzer.analyze_video(video_path, system_prompt=system_prompt)
                gr.Info("✅ Video analysis completed successfully!")
                return commentary
            except Exception as e:
                gr.Info(f"❌ Error analyzing video: {str(e)}")
                return f"Error analyzing video: {str(e)}"
        def select_gallery_video(evt: gr.SelectData):
            selected_data = evt.value
            if isinstance(selected_data, dict) and 'video' in selected_data and isinstance(selected_data['video'], dict) and 'path' in selected_data['video']:
                video_path = selected_data['video']['path']
                gr.Info(f"✅ Selected video: {os.path.basename(video_path)}")
                return video_path, int(time.time())
            elif isinstance(selected_data, tuple) and len(selected_data) > 0:
                video_path = selected_data[0]
                gr.Info(f"✅ Selected video: {os.path.basename(video_path)}")
                return video_path, int(time.time())
            elif isinstance(selected_data, str):
                gr.Info(f"✅ Selected video: {os.path.basename(selected_data)}")
                return selected_data, int(time.time())
            else:
                print(f"Warning: Unexpected data type or structure from gallery selection: {type(selected_data)}. Value: {selected_data}")
                gr.Info("❌ Failed to select video. Please try again.")
                return None, int(time.time())
        
        def select_avatar_from_gallery(evt: gr.SelectData) -> Tuple[str, str]:
            """Handle avatar gallery selection to set avatar ID and show preview."""
            selected_data = evt.value
            if isinstance(selected_data, dict) and 'caption' in selected_data:
                avatar_name = selected_data['caption']
                gr.Info(f"✅ Avatar '{avatar_name}' selected successfully!")
                return avatar_name, f"Selected avatar: {avatar_name}"
            elif isinstance(selected_data, tuple) and len(selected_data) >= 2:
                image_path, avatar_name = selected_data[0], selected_data[1]
                gr.Info(f"✅ Avatar '{avatar_name}' selected successfully!")
                return avatar_name, f"Selected avatar: {avatar_name}"
            elif isinstance(selected_data, str):
                # Extract avatar name from path
                avatar_name = os.path.basename(os.path.dirname(os.path.dirname(selected_data)))
                gr.Info(f"✅ Avatar '{avatar_name}' selected successfully!")
                return avatar_name, f"Selected avatar: {avatar_name}"
            else:
                print(f"Warning: Unexpected data type from avatar gallery selection: {type(selected_data)}. Value: {selected_data}")
                gr.Info("❌ Failed to select avatar. Please try again.")
                return "", "No avatar selected"
        
        # GPT-SoVITS Model Management Event Handlers
        if GPT_SOVITS_AVAILABLE:
            # Note: Using default models from inference_webui_fast.py
            # No model management needed - models are loaded automatically
            pass
        
        # Copy text from commentary to TTS
        def copy_text(val):
            return val
        
        def copy_to_gpt_sovits(val):
            """Copy commentary text to GPT-SoVITS target text field"""
            if GPT_SOVITS_AVAILABLE:
                return val
            else:
                return val  # Return unchanged if GPT-SoVITS not available
        
        # Gallery selection sets the video input and triggers JS to jump back to tab 1
        gallery.select(fn=select_gallery_video, outputs=[video_input, tab_switcher])

        # Client-side tab switch to the first tab when tab_switcher changes
        tab_switcher.change(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { const root = document.querySelector('#video_tabs'); if(!root) return; const first = root.querySelector('[role=tablist] button'); if(first) first.click(); }"
        )
        
        # Refresh video gallery
        refresh_gallery_btn.click(fn=load_gallery_videos, inputs=[gr.State(args.gallery_dir), gr.State(args.video_extensions)], outputs=[gallery])
        

        
        # Avatar gallery selection sets the avatar ID
        avatar_gallery.select(fn=select_avatar_from_gallery, outputs=[avatar_id_input, debug_output_info])
        
        # Refresh avatar gallery
        refresh_avatar_gallery_btn.click(fn=load_avatar_gallery, outputs=[avatar_gallery])
        # Analyze button triggers video analysis and commentary (only updates the merged box)
        analyze_btn.click(
            fn=process_analysis_video,
            inputs=[video_input, system_prompt_box],
            outputs=[commentary_tts_box]
        )
        
        # Avatar gallery selection sets the avatar ID
        avatar_gallery.select(fn=select_avatar_from_gallery, outputs=[avatar_id_input, debug_output_info])
        
        # Refresh avatar gallery
        refresh_avatar_gallery_btn.click(fn=load_avatar_gallery, outputs=[avatar_gallery])
        
        # TTS: output to driving_audio
        def tts_with_feedback(text, voice):
            if not text or not text.strip():
                gr.Info("❌ No text provided for TTS. Please enter text or analyze a video first.")
                return None
            try:
                gr.Info("🔄 Generating speech... Please wait.")
                audio_path = tts_to_audio(text, voice)
                gr.Info("✅ Speech generated successfully!")
                return audio_path
            except Exception as e:
                gr.Info(f"❌ Error generating speech: {str(e)}")
                return None
        
        tts_btn.click(
            fn=tts_with_feedback,
            inputs=[commentary_tts_box, tts_voice],
            outputs=[driving_audio]
        )
        
        # GPT-SoVITS TTS button (only if GPT-SoVITS available)
        if GPT_SOVITS_AVAILABLE:
            gpt_tts_btn.click(
                fn=lambda text, ref_audio, aux_refs, prompt_text, prompt_lang, target_lang, 
                       top_k, top_p, temp, speed_factor, batch_size, sample_steps, fragment_interval,
                       how_to_cut, super_sampling, parallel_infer, split_bucket, seed, keep_random, repetition_penalty: 
                    unified_tts_generation(
                        text, "gpt_sovits",
                        ref_audio=ref_audio, aux_refs=aux_refs, prompt_text=prompt_text, 
                        prompt_lang=prompt_lang, target_lang=target_lang,
                        top_k=top_k, top_p=top_p, temperature=temp, speed_factor=speed_factor, 
                        batch_size=batch_size, sample_steps=sample_steps, fragment_interval=fragment_interval,
                        how_to_cut=how_to_cut, super_sampling=super_sampling, parallel_infer=parallel_infer,
                        split_bucket=split_bucket, seed=seed, keep_random=keep_random, repetition_penalty=repetition_penalty
                    ),
                inputs=[
                    commentary_tts_box, gpt_ref_audio, gpt_aux_refs, gpt_prompt_text, 
                    gpt_prompt_lang, gpt_target_lang, gpt_top_k, gpt_top_p, 
                    gpt_temperature, gpt_speed_factor, gpt_batch_size, gpt_sample_steps, gpt_fragment_interval,
                    gpt_how_to_cut, gpt_super_sampling, gpt_parallel_infer, gpt_split_bucket, 
                    gpt_seed, gpt_keep_random, gpt_repetition_penalty
                ],
                outputs=[driving_audio]
            )
        # Avatar video input change (ffmpeg check)
        avatar_video_input.change(fn=check_video, inputs=[avatar_video_input], outputs=[avatar_video_input])
        # Generate Full Video (using real-time inference)
        # generate_btn.click(
        #     fn=lambda audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek: generate_standard_video(
        #         audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek,
        #         device=device, vae=vae, unet=unet, pe=pe, 
        #         weight_dtype=weight_dtype, audio_processor=audio_processor, 
        #         whisper=whisper, timesteps=timesteps,
        #         result_dir=args.inpainting_result_dir,
        #         fps=args.inpainting_fps,
        #         batch_size=args.inpainting_batch_size,
        #         output_vid_name=args.inpainting_output_vid_name,
        #         version=args.inpainting_version
        #     ),
        #     inputs=[driving_audio, avatar_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
        #     outputs=[inpainting_output_video, debug_output_info]
        # )
        
        # --- REAL-TIME AVATAR EVENT HANDLERS ---
        
        def create_avatar(avatar_id, video_path, bbox_s, extra_m, parsing_m, l_cheek, r_cheek, progress=gr.Progress(track_tqdm=True)):
            """Create a real-time avatar."""
            if not avatar_id or not avatar_id.strip():
                gr.Info("Error: Please enter an avatar ID")
                return "Error: Please enter an avatar ID"
            if not video_path:
                gr.Info("Error: Please upload a reference video")
                return "Error: Please upload a reference video"
            
            # Check if avatar already exists
            avatar_dir = os.path.join("results", "v15", "avatars", avatar_id.strip())
            if os.path.exists(avatar_dir):
                gr.Warning(f"Avatar ID '{avatar_id.strip()}' already exists! Please choose a different name.")
                return f"❌ Avatar ID '{avatar_id.strip()}' already exists! Please rename and try again."
            
            try:
                progress(0, desc="Starting avatar creation...")
                start_time = time.time()
                
                progress(0.1, desc="Loading models and preparing avatar...")
                avatar = create_realtime_avatar(
                    avatar_id=avatar_id.strip(),
                    video_path=video_path,
                    bbox_shift=bbox_s,
                    batch_size=args.inpainting_batch_size,
                    preparation=True,
                    device=device,
                    vae=vae,
                    unet=unet,
                    pe=pe,
                    weight_dtype=weight_dtype,
                    audio_processor=audio_processor,
                    whisper=whisper,
                    timesteps=timesteps,
                    version=args.inpainting_version,
                    extra_margin=extra_m,
                    parsing_mode=parsing_m,
                    left_cheek_width=l_cheek,
                    right_cheek_width=r_cheek
                )
                
                progress(1.0, desc="Avatar creation completed!")
                total_time = time.time() - start_time
                
                # Create detailed parameter information
                param_info = f"""
                Avatar '{avatar_id}' created successfully! Ready for real-time generation.

                PARAMETERS USED:
                • Avatar ID: {avatar_id.strip()}
                • Video File: {os.path.basename(video_path)}
                • BBox Shift: {bbox_s}px
                • Extra Margin: {extra_m}px  
                • Parsing Mode: {parsing_m}
                • Left Cheek Width: {l_cheek}px
                • Right Cheek Width: {r_cheek}px
                • Model Version: {args.inpainting_version}
                • Batch Size: {args.inpainting_batch_size}

                PROCESSING STATS:
                • Avatar Creation Time: {total_time:.2f}s
                • Status: Ready for real-time generation
                """
                
                gr.Info(f"Avatar '{avatar_id}' created successfully! Ready for real-time generation.")
                return param_info
            except Exception as e:
                gr.Info(f"Error creating avatar: {str(e)}")
                return f"❌ Error creating avatar: {str(e)}"
        
        def generate_realtime_video(avatar_id, audio_path, bbox_s, extra_m, parsing_m, l_cheek, r_cheek, progress=gr.Progress(track_tqdm=True)):
            """Generate real-time video using existing avatar."""
            if not avatar_id or not avatar_id.strip():
                return None, "Error: Please enter an avatar ID"
            if not audio_path:
                return None, "Error: Please provide audio for generation"
            
            try:
                progress(0, desc="Starting video generation...")
                
                progress(0.2, desc="Loading avatar and processing audio...")
                result = realtime_inference(
                    avatar_id=avatar_id.strip(),
                    audio_path=audio_path,
                    video_path="",  # Not used for real-time inference
                    bbox_shift=bbox_s,
                    batch_size=args.inpainting_batch_size,
                    preparation=False,  # Use existing avatar
                    device=device,
                    vae=vae,
                    unet=unet,
                    pe=pe,
                    weight_dtype=weight_dtype,
                    audio_processor=audio_processor,
                    whisper=whisper,
                    timesteps=timesteps,
                    version=args.inpainting_version,
                    extra_margin=extra_m,
                    parsing_mode=parsing_m,
                    left_cheek_width=l_cheek,
                    right_cheek_width=r_cheek,
                    fps=args.inpainting_fps,
                    skip_save_images=False,
                    output_vid_name=f"realtime_{avatar_id}"
                )
                
                progress(1.0, desc="Video generation completed!")
                
                # Extract timing information from result
                processing_time = result.get('processing_time', 0)
                frame_count = result.get('frame_count', 0)
                fps_achieved = result.get('fps_achieved', 0)
                output_video = result.get('output_video', None)
                
                # Create detailed parameter information
                param_info = f"""Real-time video generated successfully!

PARAMETERS USED:
• Avatar ID: {avatar_id.strip()}
• BBox Shift: {bbox_s}px
• Extra Margin: {extra_m}px  
• Parsing Mode: {parsing_m}
• Left Cheek Width: {l_cheek}px
• Right Cheek Width: {r_cheek}px
• Model Version: {args.inpainting_version}
• Target FPS: {args.inpainting_fps}
• Batch Size: {args.inpainting_batch_size}

PROCESSING STATS:
• Total Processing Time: {processing_time:.2f}s
• Frame Count: {frame_count}
• Achieved FPS: {fps_achieved:.2f}
• Audio File: {os.path.basename(audio_path)}
• Output: {output_video if output_video else 'Generated successfully'}"""
                
                return output_video, param_info
            except Exception as e:
                return None, f"❌ Error generating video: {str(e)}"
        
        # Create Avatar button
        create_avatar_btn.click(
            fn=create_avatar,
            inputs=[avatar_id_input, avatar_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[debug_output_info]
        )
        
        # Generate Real-Time Video button
        realtime_generate_btn.click(
            fn=generate_realtime_video,
            inputs=[avatar_id_input, driving_audio, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[realtime_output_video, debug_output_info]
        )
        
        # GPT-SoVITS Model Management Event Handlers
        if GPT_SOVITS_AVAILABLE:
            # Note: Using default models from inference_webui_fast.py
            # No model management needed - models are loaded automatically
            pass
        
        gr.HTML("""<style>#left-col, #right-col { padding: 1.5rem; } #video_gallery { min-height: 200px; } #avatar_gallery { min-height: 200px; margin-top: 1rem; }</style>""")
    return demo

# =================== MAIN EXECUTION ===================
def main():
    """Initializes and launches the merged Gradio application."""
    ensure_directory_exists(args.gallery_dir)
    ensure_directory_exists(args.processed_videos_dir)
    
    # Load SoVITS models on program start
    if GPT_SOVITS_AVAILABLE:
        print("🔄 Loading SoVITS models on program start...")
        try:
            # Use the path we already set up
            gpt_sovits_base = os.path.join(os.path.dirname(__file__), "GPT-SoVITS")
            gpt_sovits_path = os.path.join(gpt_sovits_base, "GPT_SoVITS")
            
            # Add to sys.path temporarily
            if gpt_sovits_path not in sys.path:
                sys.path.append(gpt_sovits_path)
            
            # Change to parent directory (GPT-SoVITS) to match the config expectations
            original_cwd = os.getcwd()
            os.chdir(gpt_sovits_base)
            
            try:
                from config import get_weights_names
                SoVITS_names, GPT_names = get_weights_names()
                print(f"✅ SoVITS models loaded: {len(SoVITS_names)} models available")
                print(f"✅ GPT models loaded: {len(GPT_names)} models available")
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"⚠️ Failed to load SoVITS models on program start: {e}")
    
    demo = merged_interface(args)
    demo.queue().launch(
        server_name=args.default_server_name,
        server_port=args.default_server_port,
        share=args.default_share,
        debug=True
    )

if __name__ == "__main__":
    main() 