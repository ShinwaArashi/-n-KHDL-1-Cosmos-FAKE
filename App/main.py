import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Scripts.scan_and_index_folder import scan_folder_and_build_index
from Scripts.search_engine import search


# Biến toàn cục để lưu FAISS index và danh sách filename
global_index = None
global_filenames = None


def load_folder(folder_path):
    global global_index, global_filenames
    if not os.path.exists(folder_path):
        return "❌ Đường dẫn không tồn tại!", []
    global_index, global_filenames = scan_folder_and_build_index(folder_path)
    return f"✅ Tải thành công {len(global_filenames)} file WAV!"

def search_audio(query_audio_path):
    if global_index is None:
        return "⚠️ Bạn cần chọn thư mục trước!", []
    results = search(query_audio_path, global_index, global_filenames)
    text = ""
    audios = []
    for i, (path, score) in enumerate(results):
        text += f"{i+1}. {os.path.basename(path)} 📏 Similarity: {score:.2f}\n"
        audios.append((path, "audio"))
    return text, audios

with gr.Blocks() as demo:
    gr.Markdown("# 🎧 COSMOS-like Sample Finder")

    with gr.Row():
        folder_input = gr.Textbox(label="📁 Đường dẫn thư mục WAV")
        load_btn = gr.Button("🔍 Tải thư mục")
    
    load_status = gr.Textbox(label="📥 Trạng thái tải thư mục")

    with gr.Row():
        audio_input = gr.Audio(label="📤 Upload WAV file", type="filepath")
        search_btn = gr.Button("🔎 Tìm kiếm")

    result_text = gr.Textbox(label="🎯 Kết quả tìm kiếm", lines=5)
    result_gallery = gr.Gallery(label="🎵 File giống nhất", columns=1)

    load_btn.click(fn=load_folder, inputs=folder_input, outputs=load_status)
    search_btn.click(fn=search_audio, inputs=audio_input, outputs=[result_text, result_gallery])

if __name__ == "__main__":
    demo.launch()
