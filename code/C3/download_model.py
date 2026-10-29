import time
import requests
from pathlib import Path

def download_visualized_bge_model(max_retries=10, retry_wait=5):
    """
    下载 Visual BGE 模型权重文件(支持断点续传与自动重试)。
    网络不稳定时会保留已下载的部分,重新运行本脚本即可续传,无需从头开始。
    """
    model_dir = Path("../../models/bge")
    model_file = model_dir / "Visualized_base_en_v1.5.pth"
    download_url = "https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth"

    model_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        # 从已下载的部分继续(断点续传)
        resume_pos = model_file.stat().st_size if model_file.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

        try:
            with requests.get(
                download_url, stream=True, headers=headers, timeout=(10, 60)
            ) as response:
                # 206 = 服务器支持续传,追加写入;200 = 忽略 Range,从头下载
                if response.status_code == 206:
                    mode, base = "ab", resume_pos
                elif response.status_code == 200:
                    mode, base = "wb", 0
                else:
                    response.raise_for_status()

                remaining = int(response.headers.get("content-length", 0))
                total_size = base + remaining

                with open(model_file, mode) as f:
                    downloaded = base
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = downloaded / total_size * 100
                                print(f"\r下载进度: {pct:.1f}% "
                                      f"({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)",
                                      end="")

            print()
            # 校验完整性
            if total_size > 0 and model_file.stat().st_size < total_size:
                raise IOError(f"文件不完整: {model_file.stat().st_size}/{total_size} 字节")

            print(f"模型下载完成: {model_file}")
            print(f"文件大小: {model_file.stat().st_size/(1024*1024):.1f} MB")
            return str(model_file)

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"\n第 {attempt}/{max_retries} 次下载失败: {e}")
            if attempt < max_retries:
                print(f"{retry_wait} 秒后重试(保留进度,支持断点续传)...")
                time.sleep(retry_wait)
            else:
                print("已达到最大重试次数。")

    # 失败时保留部分文件,下次运行可续传
    print("❌ 模型下载失败(已保留部分文件,重新运行可继续下载)")
    return None

if __name__ == "__main__":
    model_path = download_visualized_bge_model()
    if model_path:
        print(f"✅ 模型准备就绪: {model_path}")
    else:
        print("❌ 模型下载失败")
