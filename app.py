import os
import asyncio
import tempfile
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO
import re
import json

import gradio as gr
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from PIL import Image
import moviepy.editor as mp
from serpapi import GoogleSearch

# ロギング設定
# INFOレベル以上のメッセージを標準出力に出力
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数からAPIキーを取得
# Hugging Face SpacesのSecretsで設定することを強く推奨します
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

# APIキーの存在チェック
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEYが設定されていません。環境変数を確認してください。")
    # アプリケーションの起動を停止するか、機能制限を行う
    # 例: raise ValueError("GOOGLE_API_KEY is not set.")
if not SERPER_API_KEY:
    logger.error("SERPER_API_KEYが設定されていません。環境変数を確認してください。")
    # 例: raise ValueError("SERPER_API_KEY is not set.")

# Gemini 1.5 Pro設定
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    logger.info("Gemini 1.5 Proモデルの初期化に成功しました。")
except Exception as e:
    logger.critical(f"Geminiモデルの初期化中に致命的なエラーが発生しました: {e}")
    # アプリケーションの起動を停止
    raise

# サポートされる言語リスト（UIで選択可能）
SUPPORTED_LANGUAGES = {
    "日本語": "ja",
    "English": "en",
    "中文": "zh",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "한국어": "ko",
    "Русский": "ru",
    "Português": "pt",
    "العربية": "ar",
    "हिन्दी": "hi"
}

class InputProcessor:
    """ユーザー入力を処理し、言語とモダリティを検出するクラス"""
    
    @staticmethod
    async def detect_language(text: str) -> str:
        """
        テキストの言語を検出する。
        LLMにISO 639-1コードのみを応答させるよう厳密に指示します。
        """
        if not text or not text.strip(): # 空白のみの入力もチェック
            logger.debug("言語検出のため空のテキストが提供されました。デフォルト言語 'en' を返します。")
            return "en"  # デフォルト言語

        # 入力テキストのサニタイズ（LLMへのプロンプトインジェクション対策の簡易版）
        # 悪意ある文字をエスケープしたり、長さを制限したりする
        safe_text = text.strip()[:500] # 長さ制限

        try:
            response = await model.generate_content(
                f"Identify the main language of this text and respond with only the ISO 639-1 code (e.g. 'en', 'ja', 'zh', etc.). Do not include any other text, punctuation, or explanation. Text: '{safe_text}'"
            )
            # LLMの応答が想定外の場合に備え、strip()とlower()を適用し、最初の2文字を取得
            detected_lang = response.text.strip().lower()
            
            # 応答が複数の単語を含んでいたり、余分な文字が含まれていたりする可能性を考慮
            # 厳密に2文字のISOコードであることを確認
            if len(detected_lang) == 2 and detected_lang in SUPPORTED_LANGUAGES.values():
                logger.info(f"言語検出: '{safe_text[:50]}...' -> {detected_lang}")
                return detected_lang
            
            # 応答がISOコードでない場合、再度パースを試みる
            for lang_code in SUPPORTED_LANGUAGES.values():
                if lang_code in detected_lang:
                    logger.warning(f"LLMが期待外の形式で言語を検出しました ('{detected_lang}')。'{lang_code}'を抽出しました。")
                    return lang_code

            logger.warning(f"LLMが有効な言語コードを検出しませんでした: '{detected_lang}'。デフォルト言語 'en' を返します。")
            return "en"  # コードが見つからない場合はデフォルト
        except genai.types.BlockedPromptException as e:
            logger.error(f"言語検出プロンプトがブロックされました: {e}")
            return "en"
        except Exception as e:
            logger.error(f"言語検出中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return "en"  # エラーの場合はデフォルト
    
    @staticmethod
    async def process_image(image) -> Dict[str, Any]:
        """
        画像を処理し、基本的な内容把握を行う。
        PIL Imageオブジェクトを受け取ります。
        """
        if image is None:
            logger.warning("処理のため空の画像が提供されました。")
            return {
                "type": "image",
                "error": "No image provided",
                "summary": "画像は提供されませんでした"
            }

        try:
            # Gradioのgr.Image(type="pil")はPIL Imageオブジェクトを直接返すため、再変換は不要
            if not isinstance(image, Image.Image):
                logger.error(f"予期せぬ画像入力タイプ: {type(image)}。PIL.Image.Imageが必要です。")
                # 入力タイプがPIL Imageでない場合は、対応する処理を追加するかエラーを返す
                # ここではエラーを返します
                return {
                    "type": "image",
                    "error": "Invalid image input type",
                    "summary": "無効な画像入力タイプです"
                }
            
            img = image
            width, height = img.size
            format_type = img.format if hasattr(img, 'format') else "Unknown"
            
            # 画像をJPEG形式でバイトデータに変換し、base64エンコード
            # BytesIOは自動的に閉じられるため、with文は不要だが、より安全なファイルIOを意識するなら
            # 外部に保存せずメモリ上で完結させるため、この方法は適切
            buffered = BytesIO()
            # JPEG変換時のエラーを捕捉
            try:
                # 大きい画像をリサイズすることも検討（帯域幅とLLMの処理負荷軽減のため）
                # if width > 2000 or height > 2000:
                #     img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                img.save(buffered, format="JPEG", quality=90) # quality設定でファイルサイズを制御
            except Exception as e:
                logger.error(f"画像保存中のエラー: {e}", exc_info=True)
                return {
                    "type": "image",
                    "error": f"Failed to save image: {e}",
                    "summary": "画像の保存に失敗しました"
                }

            img_byte_arr = buffered.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # LLMで画像内容を解析
            # プロンプトインジェクション対策のため、ユーザーからの直接のテキストは含めない
            # ここでは内部的な指示のため問題なし
            prompt = "Describe this image concisely, focusing on the main elements, colors, and overall scene. Suggest possible topics or questions that might be related to this image. Keep the description under 150 words."
            
            try:
                response = await model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_base64}
                ])
                summary_text = response.text.strip()
            except genai.types.BlockedPromptException as e:
                logger.error(f"画像分析プロンプトがブロックされました: {e}")
                summary_text = "画像の分析は安全ポリシーによってブロックされました。"
            except Exception as e:
                logger.error(f"Geminiによる画像分析中にエラーが発生しました: {e}", exc_info=True)
                summary_text = f"画像の分析に失敗しました: {str(e)}"
            
            return {
                "type": "image",
                "metadata": {
                    "width": width,
                    "height": height,
                    "format": format_type,
                    "size_bytes": len(img_byte_arr) # 画像サイズを追加
                },
                "summary": summary_text,
                "image_data": img # 後続処理のためにPIL Imageオブジェクトを保持
            }
        except Exception as e:
            logger.error(f"画像処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return {
                "type": "image",
                "error": str(e),
                "summary": "画像の処理に失敗しました"
            }

    @staticmethod
    async def process_video(video_path: str) -> Dict[str, Any]:
        """
        動画ファイルを処理し、基本的な内容把握を行う。
        moviepyは重いため、処理時間を制限し、エラーハンドリングを強化。
        """
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"処理のため無効な動画パスが提供されました: {video_path}")
            return {
                "type": "video",
                "error": "Invalid video path or file not found",
                "summary": "無効な動画パス、またはファイルが見つかりません"
            }

        video_clip = None # 初期化
        try:
            # 動画ファイルのオープンとメタデータ取得
            # MoviePyのエラーは多岐にわたるため、ここをtry-exceptで囲む
            try:
                video_clip = mp.VideoFileClip(video_path)
                duration = video_clip.duration
                fps = video_clip.fps
                resolution = video_clip.size
                logger.info(f"動画 '{video_path}' を開きました: 期間={duration:.2f}s, FPS={fps:.2f}, 解像度={resolution}")
            except Exception as e:
                logger.error(f"動画ファイルを開く際にエラーが発生しました: {e}", exc_info=True)
                return {
                    "type": "video",
                    "error": f"Failed to open video file: {e}",
                    "summary": "動画ファイルを開けませんでした"
                }
            
            # 動画が長すぎる場合は処理を制限
            MAX_VIDEO_DURATION = 60 # 秒
            if duration > MAX_VIDEO_DURATION:
                logger.warning(f"動画が長すぎます ({duration:.2f}秒). 最初の{MAX_VIDEO_DURATION}秒のみ処理します。")
                video_clip = video_clip.subclip(0, MAX_VIDEO_DURATION)
                duration = MAX_VIDEO_DURATION # 期間も更新

            # キーフレームを抽出（冒頭、中間、終了付近）。時間を調整して重複を避ける
            num_keyframes = 3
            # durationが非常に短い場合の対策
            if duration < 1:
                keyframe_times = [0]
            elif duration < 3: # 短い動画の場合、2フレームに制限
                keyframe_times = [0, duration / 2]
            else:
                keyframe_times = [0, duration / 2, max(0, duration - 1)] # 最後のフレームに近い時間を取得
            
            keyframes = []
            for t in keyframe_times:
                try:
                    # get_frameは秒単位の時間を引数に取る
                    frame_array = video_clip.get_frame(t)
                    img = Image.fromarray(frame_array)
                    
                    buffered = BytesIO()
                    # 圧縮品質を設定してファイルサイズを抑制
                    img.save(buffered, format="JPEG", quality=80) 
                    img_byte_arr = buffered.getvalue()
                    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                    keyframes.append({
                        "time": t,
                        "image_base64": img_base64,
                        "image": img # 後続処理のためにPIL Imageオブジェクトを保持
                    })
                    logger.debug(f"キーフレームを {t:.2f}s で抽出しました。")
                except Exception as e:
                    logger.warning(f"キーフレーム {t:.2f}s の抽出中にエラーが発生しました: {e}", exc_info=True)
                    continue # 特定のフレームが失敗しても続行
            
            if not keyframes:
                logger.warning("動画からキーフレームを抽出できませんでした。")
                return {
                    "type": "video",
                    "error": "Failed to extract keyframes",
                    "summary": "動画のキーフレームを抽出できませんでした"
                }

            # 音声がある場合は簡易的に処理 (リソース制約のため、詳細な分析はスキップ)
            audio_summary = "音声なし"
            if video_clip.audio:
                audio_summary = "音声あり（詳細な処理はリソース制約のためスキップ）"
                logger.debug("動画に音声トラックが存在します。")
            
            # キーフレームを使って動画の内容を分析
            prompt = """Analyze this video based on the provided key frames. Focus on:
            1. The main actions, events, or objects visible.
            2. Any narrative progression or changes across the frames.
            3. The overall context or theme of the video.
            4. Potential questions or topics that might be relevant.
            Keep the description concise and under 200 words.
            """
            content_parts = [prompt]
            
            for frame in keyframes:
                content_parts.append({
                    "mime_type": "image/jpeg", 
                    "data": frame["image_base64"]
                })
            
            summary_text = "動画の分析に失敗しました。"
            try:
                response = await model.generate_content(content_parts)
                summary_text = response.text.strip()
            except genai.types.BlockedPromptException as e:
                logger.error(f"動画分析プロンプトがブロックされました: {e}")
                summary_text = "動画の分析は安全ポリシーによってブロックされました。"
            except Exception as e:
                logger.error(f"Geminiによる動画分析中にエラーが発生しました: {e}", exc_info=True)
                summary_text = f"動画の分析に失敗しました: {str(e)}"
            
            return {
                "type": "video",
                "metadata": {
                    "duration": duration,
                    "fps": fps,
                    "resolution": resolution,
                    "frame_count": len(keyframes)
                },
                "keyframes": keyframes, # PIL Imageオブジェクトも含まれる
                "audio_summary": audio_summary,
                "summary": summary_text
            }
        except Exception as e:
            logger.error(f"動画処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return {
                "type": "video",
                "error": str(e),
                "summary": "動画の処理に失敗しました"
            }
        finally:
            if video_clip:
                video_clip.close() # 動画リソースを確実に解放

    @staticmethod
    async def process_input(text: str, image=None, video=None) -> Dict[str, Any]:
        """
        すべての入力を処理し、コンテキストとして統合する。
        各モダリティの処理は並行して実行されます。
        """
        context = {"input_type": [], "original_text": text} # オリジナルテキストを保持
        tasks = []
        
        # テキスト処理タスク
        if text and text.strip():
            context["input_type"].append("text")
            # detect_languageはテキストのみに依存するため、ここでタスクに追加
            tasks.append(InputProcessor.detect_language(text))
        else:
            logger.debug("テキスト入力がありませんでした。")
            context["detected_language"] = "en" # テキストがない場合はデフォルト言語を設定

        # 画像処理タスク
        if image is not None:
            context["input_type"].append("image")
            tasks.append(InputProcessor.process_image(image))
        else:
            logger.debug("画像入力がありませんでした。")

        # 動画処理タスク
        if video is not None:
            context["input_type"].append("video")
            # videoがGradioのTemporaryFileパスであると想定
            tasks.append(InputProcessor.process_video(video))
        else:
            logger.debug("動画入力がありませんでした。")

        # 全てのタスクを並列実行し、結果を待つ
        results = await asyncio.gather(*tasks, return_exceptions=True) # エラーが発生しても他のタスクが完了する

        # 結果をコンテキストに統合
        result_index = 0
        if "text" in context["input_type"]:
            lang_result = results[result_index]
            if isinstance(lang_result, Exception):
                logger.error(f"言語検出タスクでエラー: {lang_result}")
                context["detected_language"] = "en" # エラー時はデフォルト
            else:
                context["detected_language"] = lang_result
            result_index += 1
        
        if "image" in context["input_type"]:
            img_analysis_result = results[result_index]
            if isinstance(img_analysis_result, Exception):
                logger.error(f"画像処理タスクでエラー: {img_analysis_result}")
                context["image_analysis"] = {"type": "image", "error": str(img_analysis_result), "summary": "画像処理中にエラーが発生しました"}
            else:
                context["image_analysis"] = img_analysis_result
            result_index += 1
        
        if "video" in context["input_type"]:
            video_analysis_result = results[result_index]
            if isinstance(video_analysis_result, Exception):
                logger.error(f"動画処理タスクでエラー: {video_analysis_result}")
                context["video_analysis"] = {"type": "video", "error": str(video_analysis_result), "summary": "動画処理中にエラーが発生しました"}
            else:
                context["video_analysis"] = video_analysis_result
        
        logger.info(f"最終入力コンテキスト構築完了: {context.keys()}")
        return context

class TaskDecomposer:
    """ユーザーのタスクをサブタスクに分解するクラス"""
    
    @staticmethod
    async def decompose_task(context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        コンテキストからサブタスクに分解する。
        LLMの出力形式を厳密に制御するためのプロンプトを追加し、パースを強化します。
        """
        input_types = context.get("input_type", [])
        original_text = context.get("original_text", "")
        detected_language = context.get("detected_language", "en")

        prompt_parts = [
            "You are an expert task decomposer. Decompose the following user request into specific, independent subtasks that can be executed in parallel.",
            "Each subtask must be a JSON object with the following keys:",
            "- `title`: A concise title for the subtask (string).",
            "- `description`: A detailed description of what the subtask should achieve (string).",
            "- `optimal_language`: The ISO 639-1 code (e.g., 'en', 'ja') for the best language to perform web searches for this subtask. This should ideally match the content of the search queries.",
            "- `search_queries`: A JSON array of 1-3 highly effective, concise search queries (strings) to find relevant information. Empty array if no web search is needed.",
            "- `requires_web_search`: A boolean indicating if this subtask absolutely requires web search.",
            "If the request is simple (e.g., just asking a question), decompose it into a single primary web search task.",
            "If no explicit query is given but media is provided, analyze the media and suggest relevant search queries based on its content.",
            "Do NOT include any explanatory text or preamble in your response. Output ONLY the JSON list."
        ]
        
        # テキスト入力の追加
        if original_text:
            prompt_parts.append(f"User request: {original_text}")
        
        # 画像分析結果の追加
        if "image" in input_types and "image_analysis" in context:
            img_summary = context['image_analysis'].get('summary', 'No analysis available')
            if img_summary and "error" not in context['image_analysis']:
                prompt_parts.append(f"The user has uploaded an image. Initial analysis: {img_summary}")
        
        # 動画分析結果の追加
        if "video" in input_types and "video_analysis" in context:
            video_summary = context['video_analysis'].get('summary', 'No analysis available')
            if video_summary and "error" not in context['video_analysis']:
                prompt_parts.append(f"The user has uploaded a video. Initial analysis: {video_summary}")

        if not original_text and not ("image" in input_types or "video" in input_types):
             prompt_parts.append("The user provided no specific query or media. Assume a general knowledge task.")

        full_prompt = "\n".join(prompt_parts)
        logger.debug(f"タスク分解プロンプト: {full_prompt}")

        try:
            response = await model.generate_content(full_prompt)
            subtasks_text = response.text.strip()
            logger.debug(f"LLMからのタスク分解応答: {subtasks_text}")
            
            # JSONの部分を抽出 - LLMが余分なテキストを含める可能性に備える
            json_match = re.search(r'\[\s*\{.*\}\s*\]', subtasks_text, re.DOTALL)
            if json_match:
                subtasks_json = json_match.group(0)
                subtasks = json.loads(subtasks_json)
                
                # スキーマ検証 (簡易版)
                for task in subtasks:
                    if not all(k in task for k in ["title", "description", "optimal_language", "search_queries", "requires_web_search"]):
                        raise ValueError("LLM応答のサブタスクに必須キーが不足しています。")
                    if not isinstance(task["search_queries"], list):
                        raise ValueError("search_queriesがリストではありません。")
                logger.info(f"タスク分解成功。生成されたサブタスク数: {len(subtasks)}")
            else:
                logger.warning("LLM応答からJSONがパースできませんでした。再フォーマットを試みます。")
                # JSON形式でない場合、再フォーマットを試みる
                reformatting_prompt = f"The following output was intended to be a JSON list of objects, but it's malformed. Please convert it into a proper JSON list format with the specified schema (title, description, optimal_language, search_queries, requires_web_search):\n\n{subtasks_text}"
                reformat_response = await model.generate_content(reformatting_prompt)
                reformatted_text = reformat_response.text.strip()
                logger.debug(f"LLMからの再フォーマット応答: {reformatted_text}")
                json_match = re.search(r'\[\s*\{.*\}\s*\]', reformatted_text, re.DOTALL)
                
                if json_match:
                    subtasks_json = json_match.group(0)
                    subtasks = json.loads(subtasks_json)
                    for task in subtasks: # 再度スキーマ検証
                        if not all(k in task for k in ["title", "description", "optimal_language", "search_queries", "requires_web_search"]):
                            raise ValueError("再フォーマットされたLLM応答のサブタスクに必須キーが不足しています。")
                        if not isinstance(task["search_queries"], list):
                            raise ValueError("再フォーマットされたsearch_queriesがリストではありません。")
                    logger.info(f"タスク分解成功（再フォーマット後）。生成されたサブタスク数: {len(subtasks)}")
                else:
                    logger.error("JSONパースおよび再フォーマットに失敗しました。デフォルトタスクを生成します。")
                    raise ValueError("JSON parse and reformatting failed.")
            
            # コンテキストにメディア分析タスクを追加（重複しないように）
            # これらのタスクはLLMが生成したものではなく、システムが必ず実行するもの
            # 優先的に処理されるようリストの先頭に追加
            system_generated_media_tasks = []
            if "image" in input_types and context.get("image_analysis") and "error" not in context["image_analysis"]:
                system_generated_media_tasks.append({
                    "title": "Image Content Analysis",
                    "description": "Analyze the uploaded image in detail using the LLM.",
                    "optimal_language": "en", # LLMへの指示は英語が最適
                    "search_queries": [],
                    "requires_web_search": False,
                    "media_type": "image_detail_analysis" # 特定の処理をトリガーするカスタムキー
                })
            
            if "video" in input_types and context.get("video_analysis") and "error" not in context["video_analysis"]:
                system_generated_media_tasks.append({
                    "title": "Video Content Analysis",
                    "description": "Analyze the uploaded video keyframes in detail using the LLM.",
                    "optimal_language": "en", # LLMへの指示は英語が最適
                    "search_queries": [],
                    "requires_web_search": False,
                    "media_type": "video_detail_analysis" # 特定の処理をトリガーするカスタムキー
                })
            
            subtasks = system_generated_media_tasks + subtasks
            
            return subtasks
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"タスク分解のJSONパースまたはスキーマ検証に失敗しました: {e}", exc_info=True)
            return TaskDecomposer._generate_default_tasks(context, original_text, detected_language, input_types)
        except genai.types.BlockedPromptException as e:
            logger.error(f"タスク分解プロンプトが安全ポリシーによってブロックされました: {e}")
            return TaskDecomposer._generate_default_tasks(context, original_text, detected_language, input_types, 
                                                            error_message="タスク分解は安全ポリシーによってブロックされました。")
        except Exception as e:
            logger.error(f"タスク分解中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return TaskDecomposer._generate_default_tasks(context, original_text, detected_language, input_types)

    @staticmethod
    def _generate_default_tasks(context: Dict[str, Any], original_text: str, detected_language: str, input_types: List[str], error_message: Optional[str] = None) -> List[Dict[str, Any]]:
        """エラー発生時にデフォルトのサブタスクを生成するヘルパー関数"""
        logger.warning("タスク分解に失敗しました。デフォルトタスクを生成します。")
        
        # エラーメッセージがあればそれを使用
        query_text = error_message if error_message else original_text
        
        # テキストがない場合、メディア分析の要約をクエリとして使用
        if not query_text:
            if "image" in input_types and "image_analysis" in context and "summary" in context["image_analysis"]:
                query_text = context["image_analysis"]["summary"][:200]
            elif "video" in input_types and "video_analysis" in context and "summary" in context["video_analysis"]:
                query_text = context["video_analysis"]["summary"][:200]
            else:
                query_text = "general information" # fallback

        # デフォルトのWeb検索タスク
        default_tasks = [{
            "title": "Main Information Search",
            "description": "Execute web search based on the primary user input or detected media content.",
            "optimal_language": detected_language,
            "search_queries": [query_text[:200]] if query_text else ["general knowledge"],
            "requires_web_search": True
        }]
        
        # デフォルトタスクにメディアの詳細分析タスクを追加
        if "image" in input_types and context.get("image_analysis") and "error" not in context["image_analysis"]:
             default_tasks.insert(0, { # 先頭に挿入
                "title": "Image Content Analysis",
                "description": "Analyze the uploaded image in detail using the LLM.",
                "optimal_language": "en",
                "search_queries": [],
                "requires_web_search": False,
                "media_type": "image_detail_analysis"
            })
        
        if "video" in input_types and context.get("video_analysis") and "error" not in context["video_analysis"]:
            default_tasks.insert(0, { # 先頭に挿入
                "title": "Video Content Analysis",
                "description": "Analyze the uploaded video keyframes in detail using the LLM.",
                "optimal_language": "en",
                "search_queries": [],
                "requires_web_search": False,
                "media_type": "video_detail_analysis"
            })
        
        return default_tasks


class WebResearcherAndAnalyzer:
    """Webからの情報収集と分析を行うクラス"""
    
    @staticmethod
    async def search_web(query: str, lang: str = "en", num_results: int = 5) -> List[Dict[str, Any]]:
        """
        指定されたクエリでWeb検索を実行し、結果を返す。
        SerpAPIを使用し、エラーハンドリングを強化。
        """
        if not SERPER_API_KEY:
            logger.error("SERPER_API_KEYが設定されていないため、Web検索を実行できません。")
            return [{
                "title": "API Key Error",
                "link": "",
                "snippet": "Serper APIキーが設定されていません。",
                "source": "error"
            }]
        
        if not query or not query.strip():
            logger.warning("空のWeb検索クエリが提供されました。")
            return [{
                "title": "Empty Query",
                "link": "",
                "snippet": "検索クエリが空です。",
                "source": "error"
            }]

        try:
            # SerpAPIを使用してWeb検索を実行
            params = {
                "q": query,
                "api_key": SERPER_API_KEY,
                "num": min(num_results, 10), # 最大結果数を10に制限（APIとリソースの制約）
                "hl": lang,  # 検索言語設定
                "gl": "us"   # 地域設定（デフォルトは米国、必要に応じて動的に変更可能）
            }
            
            logger.info(f"Web検索開始: クエリ='{query[:50]}...', 言語='{lang}'")
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # SerpAPIからのエラー応答をチェック
            if "error" in results:
                logger.error(f"SerpAPIからエラー応答: {results['error']}")
                return [{
                    "title": "SerpAPI Error",
                    "link": "",
                    "snippet": f"SerpAPIからのエラー: {results['error']}",
                    "source": "error"
                }]

            organic_results = results.get("organic_results", []) # SerpAPIは'organic_results'を返す
            processed_results = []
            
            for result in organic_results:
                processed_results.append({
                    "title": result.get("title", "No title"),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "No description available"),
                    "source": "web_search"
                })
            
            logger.info(f"Web検索完了: '{query[:50]}...'、取得結果数: {len(processed_results)}")
            return processed_results
        except requests.exceptions.RequestException as e:
            logger.error(f"Web検索中のネットワークエラー: {e}", exc_info=True)
            return [{
                "title": "Network Error",
                "link": "",
                "snippet": f"ネットワーク接続エラーが発生しました: {str(e)}",
                "source": "error"
            }]
        except Exception as e:
            logger.error(f"Web検索中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return [{
                "title": "Search Error",
                "link": "",
                "snippet": f"検索中に予期せぬエラーが発生しました: {str(e)}",
                "source": "error"
            }]
    
    @staticmethod
    async def fetch_webpage_content(url: str, max_chars: int = 15000) -> str:
        """
        指定されたURLからWebページの内容を取得し、テキストコンテンツを抽出する。
        より厳密なURL検証とタイムアウト、サイズ制限を追加。
        """
        if not url or not re.match(r'^https?://', url): # URLの基本的な形式を検証
            logger.warning(f"無効なURLが提供されました: {url}")
            return "コンテンツの取得に失敗しました: 無効なURLです。"

        # セキュリティ: リダイレクトを制限する (デフォルトはTrue, max_redirectsを設定することで安全性を高める)
        # requests-html や httpx の方が非同期には向いているが、requestsで続ける場合は注意
        try:
            # タイムアウトを短めに設定し、巨大な応答を避けるためにstream=Trueを使用することも検討
            # しかし、現在のrequests.getはデフォルトで全てのコンテンツをメモリにロードする
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # 非同期でrequestsを呼び出すには、loop.run_in_executorを使う必要があります
            # ここでは同期的なrequests.getをasyncio.to_threadでラップ
            response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15) # タイムアウトを15秒に設定
            response.raise_for_status() # 200以外のステータスコードでHTTPErrorを発生させる

            # レスポンスサイズが大きすぎないかチェック（メモリ枯渇を防ぐ）
            if len(response.content) > 5 * 1024 * 1024: # 5MBを超える場合は処理しない
                logger.warning(f"Webページが大きすぎます（{len(response.content)}バイト）。URL: {url}")
                return "コンテンツの取得に失敗しました: Webページが大きすぎます。"

            # BeautifulSoupでHTMLを解析
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # スクリプト、スタイル、メタタグ、ナビゲーション、フッターなどを削除して主要コンテンツに集中
            for tag in soup(["script", "style", "meta", "noscript", "iframe", "nav", "footer", "header", "aside"]):
                tag.extract()
            
            # テキストを抽出
            text = soup.get_text(separator=' ', strip=True)
            
            # テキストを整形（複数の空白を一つにまとめる）
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 文字数制限
            if len(text) > max_chars:
                logger.warning(f"Webコンテンツが長すぎます ({len(text)}文字)。切り捨てました。URL: {url}")
                text = text[:max_chars] + "... (content truncated)"
            
            logger.info(f"Webページコンテンツ取得完了: URL='{url}', 文字数={len(text)}")
            return text
        except requests.exceptions.Timeout:
            logger.error(f"Webページの取得がタイムアウトしました ({url})", exc_info=True)
            return "コンテンツの取得に失敗しました: タイムアウト。"
        except requests.exceptions.TooManyRedirects:
            logger.error(f"Webページの取得がリダイレクトしすぎました ({url})", exc_info=True)
            return "コンテンツの取得に失敗しました: リダイレクトが多すぎます。"
        except requests.exceptions.RequestException as e:
            logger.error(f"Webページの取得中にネットワークまたはHTTPエラーが発生しました ({url}): {e}", exc_info=True)
            return f"コンテンツの取得に失敗しました: ネットワークまたはHTTPエラー: {str(e)}"
        except Exception as e:
            logger.error(f"Webページの取得中に予期せぬエラーが発生しました ({url}): {e}", exc_info=True)
            return f"コンテンツの取得に失敗しました: 予期せぬエラー: {str(e)}"
    
    @staticmethod
    async def analyze_media_content(context: Dict[str, Any], media_type: str) -> Dict[str, Any]:
        """
        アップロードされたメディア（画像/動画）をLLMを使って詳細に分析する。
        InputProcessorで一度分析された内容を基に、より詳細なプロンプトを生成します。
        """
        analysis_data = {}
        if media_type == "image_detail_analysis" and "image_analysis" in context:
            analysis_data = context["image_analysis"]
            original_media_type = "image"
        elif media_type == "video_detail_analysis" and "video_analysis" in context:
            analysis_data = context["video_analysis"]
            original_media_type = "video"
        else:
            return {"type": "error", "message": f"不明または無効なメディア分析タイプ: {media_type}"}

        if "error" in analysis_data:
            logger.warning(f"以前の{original_media_type}処理でエラーがあったため、詳細分析をスキップします。")
            return {
                "type": f"{original_media_type}_detailed_analysis_skipped",
                "summary": analysis_data.get("summary", "エラーにより詳細分析スキップ"),
                "detailed_analysis": "以前の処理でエラーが発生したため、詳細分析は実行されませんでした。",
                "metadata": analysis_data.get("metadata", {})
            }

        try:
            if original_media_type == "image":
                img = analysis_data.get("image_data")
                if not img:
                    logger.error("画像詳細分析のためにPIL Imageデータが見つかりません。")
                    return {"type": "image_detailed_analysis_error", "message": "画像データが見つかりません。"}
                
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=90)
                img_byte_arr = buffered.getvalue()
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                prompt = """Provide a detailed and comprehensive analysis of this image. Address the following aspects:
                1.  **Main Elements**: Identify and describe all prominent objects, people, animals, and their relative positions.
                2.  **Context and Scene**: Describe the setting, environment, and overall atmosphere. Is it indoor/outdoor? Day/night?
                3.  **Colors and Lighting**: Comment on the dominant colors, lighting conditions, and how they contribute to the mood.
                4.  **Action/Activity**: If there is any action, describe what is happening.
                5.  **Text in Image**: Transcribe and explain any visible text.
                6.  **Potential Implications/Interpretation**: What message might this image convey? What is its likely purpose or origin?
                7.  **Suggested Follow-up Questions/Topics**: Based on your analysis, what are good related questions or topics for further research?
                
                Format your response clearly with headings for each section. Be thorough but concise.
                """
                content_parts = [prompt, {"mime_type": "image/jpeg", "data": img_base64}]
                
            elif original_media_type == "video":
                keyframes = analysis_data.get("keyframes", [])
                if not keyframes:
                    logger.error("動画詳細分析のためにキーフレームデータが見つかりません。")
                    return {"type": "video_detailed_analysis_error", "message": "キーフレームデータが見つかりません。"}
                
                prompt = """Provide a detailed and comprehensive analysis of this video based on the provided key frames. Address the following aspects:
                1.  **Main Progression/Narrative**: Describe the sequence of events or changes observed across the keyframes. What is the overall story or activity?
                2.  **Key Elements per Frame**: Identify and describe important objects, people, or settings in each frame.
                3.  **Emotional Tone/Atmosphere**: What is the general mood or feeling conveyed by the video?
                4.  **Context and Purpose**: What is the likely context or purpose of this video? (e.g., news, personal, advertisement, educational).
                5.  **Audio Impression**: Based on the provided audio summary, how might audio contribute to the video's content? (Note: no detailed audio analysis performed).
                6.  **Suggested Follow-up Questions/Topics**: Based on your analysis, what are good related questions or topics for further research?
                
                Format your response clearly with headings for each section. Be thorough but concise.
                """
                content_parts = [prompt]
                for frame in keyframes:
                    if "image_base64" in frame:
                        content_parts.append({"mime_type": "image/jpeg", "data": frame["image_base64"]})

            detailed_analysis_text = "詳細分析を生成できませんでした。"
            try:
                response = await model.generate_content(content_parts)
                detailed_analysis_text = response.text.strip()
            except genai.types.BlockedPromptException as e:
                logger.error(f"詳細メディア分析プロンプトがブロックされました: {e}")
                detailed_analysis_text = "詳細なメディア分析は安全ポリシーによってブロックされました。"
            except Exception as e:
                logger.error(f"Geminiによる詳細メディア分析中にエラーが発生しました: {e}", exc_info=True)
                detailed_analysis_text = f"詳細メディア分析に失敗しました: {str(e)}"

            return {
                "type": f"{original_media_type}_detailed_analysis",
                "summary": analysis_data.get("summary", ""),
                "detailed_analysis": detailed_analysis_text,
                "metadata": analysis_data.get("metadata", {}),
                "audio_summary": analysis_data.get("audio_summary", "") if original_media_type == "video" else None
            }
        except Exception as e:
            logger.error(f"詳細メディア分析中に予期せぬエラーが発生しました ({media_type}): {e}", exc_info=True)
            return {
                "type": f"{original_media_type}_detailed_analysis_error",
                "error": str(e),
                "message": f"{original_media_type}の詳細分析中にエラーが発生しました"
            }


class ParallelTaskExecutor:
    """サブタスクを並列実行するクラス"""
    
    @staticmethod
    async def execute_subtask(subtask: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """単一のサブタスクを実行する"""
        start_time = time.time()
        results = []
        task_type = "unknown"
        
        try:
            # メディア詳細分析タスク
            if subtask.get("media_type") in ["image_detail_analysis", "video_detail_analysis"]:
                task_type = subtask["media_type"]
                logger.info(f"タスク実行: {subtask['title']} ({task_type})")
                media_result = await WebResearcherAndAnalyzer.analyze_media_content(
                    context, 
                    subtask["media_type"]
                )
                results.append(media_result)
            
            # Web検索を必要とするタスク
            elif subtask.get("requires_web_search", False) and subtask.get("search_queries"):
                task_type = "web_search"
                logger.info(f"タスク実行: {subtask['title']} ({task_type}) with queries: {subtask['search_queries']}")
                search_results = []
                
                # 各検索クエリを処理
                query_tasks = []
                for query in subtask["search_queries"]:
                    # 検索クエリのサニタイズ (簡易版)
                    safe_query = query.strip()[:200]
                    if safe_query:
                        query_tasks.append(WebResearcherAndAnalyzer.search_web(
                            safe_query, 
                            lang=subtask.get("optimal_language", "en"),
                            num_results=3 # リソース制約のため検索結果を制限
                        ))
                    else:
                        logger.warning(f"無効な検索クエリがスキップされました: '{query}'")
                
                if query_tasks:
                    all_query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
                    for res in all_query_results:
                        if isinstance(res, Exception):
                            logger.error(f"個別の検索クエリ実行中にエラー: {res}", exc_info=True)
                            results.append({"title": "Sub-query Error", "snippet": str(res), "source": "error"})
                        else:
                            search_results.extend(res)
                
                # 検索結果のうち上位のみを選択してWebページコンテンツを取得
                # 重複リンクを排除し、処理するURL数を制限
                unique_links = set()
                top_results_to_fetch = []
                for sr in search_results:
                    if sr.get("link") and sr["link"] not in unique_links:
                        unique_links.add(sr["link"])
                        top_results_to_fetch.append(sr)
                    if len(top_results_to_fetch) >= 3: # 最大3つのリンクからコンテンツを取得
                        break
                
                fetch_tasks = []
                for result_item in top_results_to_fetch:
                    if result_item.get("link"):
                        fetch_tasks.append(WebResearcherAndAnalyzer.fetch_webpage_content(result_item["link"]))
                
                if fetch_tasks:
                    web_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    
                    for i, content_result in enumerate(web_contents):
                        if i < len(top_results_to_fetch):
                            current_result_item = top_results_to_fetch[i]
                            if isinstance(content_result, Exception):
                                logger.error(f"Webコンテンツ取得中にエラーが発生: {content_result}", exc_info=True)
                                current_result_item["content_summary"] = f"コンテンツ取得エラー: {str(content_result)}"
                                current_result_item["error"] = True
                            else:
                                # コンテンツの長さが十分にない場合は要約をスキップ
                                if len(content_result) > 200: # 簡潔な要約のために200文字を閾値に
                                    summary_prompt = f"""Summarize the following web content from {current_result_item.get('link', 'unknown link')} in 3-5 sentences, focusing on the most relevant information for the task: "{subtask.get('title')}".
                                    
                                    WEB CONTENT:
                                    {content_result[:15000]}  # 長すぎるコンテンツは切り捨て
                                    """
                                    try:
                                        summary_response = await model.generate_content(summary_prompt)
                                        current_result_item["content_summary"] = summary_response.text.strip()
                                    except genai.types.BlockedPromptException as e:
                                        logger.error(f"コンテンツ要約プロンプトがブロックされました: {e}")
                                        current_result_item["content_summary"] = "コンテンツ要約は安全ポリシーによってブロックされました。"
                                    except Exception as e:
                                        logger.error(f"コンテンツ要約中にエラー: {e}", exc_info=True)
                                        current_result_item["content_summary"] = "要約を生成できませんでした。"
                                else:
                                    current_result_item["content_summary"] = content_result # 短い場合はそのまま格納
                                
                            results.append(current_result_item) # 各Web検索結果と要約を追加
                else:
                    logger.info(f"クエリ '{query[:50]}...' のWeb検索結果から取得する有効なリンクがありませんでした。")
                    results.append({"title": f"No valid links for '{query[:50]}...'", "snippet": "Web検索結果から有効なコンテンツが見つかりませんでした。", "source": "no_content"})

            else:
                logger.warning(f"未知のサブタスクタイプまたは要件が満たされませんでした: {subtask.get('title')}")
                results.append({
                    "title": "Unrecognized Task",
                    "description": subtask.get("description", "Unknown"),
                    "snippet": "このタスクは処理されませんでした。",
                    "source": "internal_error"
                })

            # 実行時間を計測
            execution_time = time.time() - start_time
            logger.info(f"サブタスク '{subtask['title']}' 実行完了。時間: {execution_time:.2f}s")

            return {
                "subtask_title": subtask["title"],
                "description": subtask.get("description", ""),
                "results": results,
                "execution_time": execution_time,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"サブタスク '{subtask.get('title', 'Unknown')}' 実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return {
                "subtask_title": subtask.get("title", "Unknown"),
                "error": str(e),
                "results": [],
                "execution_time": time.time() - start_time,
                "status": "failed"
            }
    
    @staticmethod
    async def execute_tasks(subtasks: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """すべてのサブタスクを並列実行する"""
        if not subtasks:
            logger.warning("実行するサブタスクがありません。")
            return []
        
        tasks = [ParallelTaskExecutor.execute_subtask(subtask, context) for subtask in subtasks]
        
        # すべてのタスクの結果を待つ。個別のタスクのエラーはexecute_subtask内で処理済み
        results = await asyncio.gather(*tasks)
        
        return results

class ResultSynthesizer:
    """タスク実行結果を統合して最終出力を生成するクラス"""
    
    @staticmethod
    async def synthesize_results(
        task_results: List[Dict[str, Any]], 
        context: Dict[str, Any],
        output_language: str
    ) -> str:
        """
        タスク結果を統合して指定された言語で最終結果を生成する。
        より構造化された出力とエラーレポートを含めます。
        """
        if not task_results:
            logger.warning("統合するタスク結果がありません。")
            return f"申し訳ありませんが、情報を取得できませんでした。({output_language}で回答)"

        synthesis_prompt_parts = [
            f"Based on the following information, create a comprehensive and well-structured response in {output_language}.",
            "Prioritize the most relevant information and directly answer the user's original query. Use markdown for clear formatting, including headings and bullet points where appropriate.",
            "If any media (image/video) was provided, integrate its analysis into the response.",
            "If web search was performed, cite the sources by including the title and link of relevant results in a 'Sources' section at the end.",
            "If there were any issues or errors in processing (e.g., failed searches, blocked content), mention them transparently but concisely.",
            "\n---",
            f"**User's Original Query**: {context.get('original_text', 'N/A')}"
        ]
        
        # アップロードされたメディアの情報を追加
        if "image" in context.get("input_type", []):
            synthesis_prompt_parts.append("\n**Uploaded Image Information**:")
            img_analysis = context.get("image_analysis", {})
            if "error" in img_analysis:
                synthesis_prompt_parts.append(f"  - Image processing failed: {img_analysis['summary']}")
            else:
                synthesis_prompt_parts.append(f"  - Initial Image Summary: {img_analysis.get('summary', 'No summary available')}")
        
        if "video" in context.get("input_type", []):
            synthesis_prompt_parts.append("\n**Uploaded Video Information**:")
            video_analysis = context.get("video_analysis", {})
            if "error" in video_analysis:
                synthesis_prompt_parts.append(f"  - Video processing failed: {video_analysis['summary']}")
            else:
                synthesis_prompt_parts.append(f"  - Initial Video Summary: {video_analysis.get('summary', 'No summary available')}")
        
        synthesis_prompt_parts.append("\n---")
        synthesis_prompt_parts.append("\n**Task Execution Results**:")
        
        collected_sources = []
        
        for i, task_result in enumerate(task_results):
            synthesis_prompt_parts.append(f"\n### {task_result.get('subtask_title', f'Task {i+1}')}")
            synthesis_prompt_parts.append(f"Description: {task_result.get('description', 'N/A')}")
            
            if task_result.get("status") == "failed":
                synthesis_prompt_parts.append(f"**Status**: Failed. Error: {task_result['error']}")
                continue
            
            results = task_result.get("results", [])
            
            if not results:
                synthesis_prompt_parts.append("No specific results obtained for this task.")
            
            for j, result in enumerate(results):
                if isinstance(result, dict):
                    if result.get("type") == "image_detailed_analysis":
                        synthesis_prompt_parts.append(f"\n#### Detailed Image Analysis:")
                        synthesis_prompt_parts.append(result.get("detailed_analysis", "詳細な画像分析は提供されていません。"))
                    elif result.get("type") == "video_detailed_analysis":
                        synthesis_prompt_parts.append(f"\n#### Detailed Video Analysis:")
                        synthesis_prompt_parts.append(result.get("detailed_analysis", "詳細な動画分析は提供されていません。"))
                        if result.get("audio_summary"):
                            synthesis_prompt_parts.append(f"Audio Summary: {result['audio_summary']}")
                    elif result.get("source") == "web_search":
                        # Web検索結果と要約
                        synthesis_prompt_parts.append(f"\n- **{result.get('title', 'N/A')}**")
                        synthesis_prompt_parts.append(f"  Link: <{result.get('link', '#')}>") # Markdownリンク形式
                        if "content_summary" in result:
                            synthesis_prompt_parts.append(f"  Summary: {result['content_summary']}")
                        else:
                            synthesis_prompt_parts.append(f"  Snippet: {result.get('snippet', 'No snippet available')}")
                        
                        # 引用のための情報収集
                        if result.get('link'):
                            collected_sources.append({"title": result.get('title', 'N/A'), "link": result['link']})
                    elif result.get("source") == "error":
                        synthesis_prompt_parts.append(f"\n- **Error in search result**: {result.get('title', 'N/A')}")
                        synthesis_prompt_parts.append(f"  Details: {result.get('snippet', 'No error details.')}")
                else:
                    synthesis_prompt_parts.append(f"- Unstructured result: {str(result)}")

        # 収集した引用情報を追加
        if collected_sources:
            synthesis_prompt_parts.append("\n---\n**Sources (情報源)**:")
            for idx, source in enumerate(collected_sources):
                synthesis_prompt_parts.append(f"{idx+1}. [{source['title']}]({source['link']})")
        
        final_prompt = "\n".join(synthesis_prompt_parts)
        logger.debug("最終合成プロンプト（一部）:\n" + final_prompt[:1000] + "...") # プロンプトが長い場合があるため一部のみログ

        try:
            # 最終応答生成
            final_response = await model.generate_content(final_prompt)
            final_text = final_response.text.strip()
            logger.info("最終応答の生成に成功しました。")
            return final_text
        except genai.types.BlockedPromptException as e:
            logger.error(f"最終合成プロンプトが安全ポリシーによってブロックされました: {e}")
            return f"申し訳ありませんが、最終的な応答は安全ポリシーによってブロックされました。提供された情報に基づいて回答を生成できません。 ({output_language}で回答)"
        except Exception as e:
            logger.error(f"最終応答の生成中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return f"申し訳ありませんが、結果の統合中にエラーが発生しました: {str(e)}。詳細についてはログをご確認ください。 ({output_language}で回答)"


# Gradio UI Example (for app.py) - この部分はプロジェクトのapp.pyに配置してください
# -----------------------------------------------------------------------------
async def run_agent(user_query: str, input_image: Image.Image, input_video: str, output_lang_display: str, progress=gr.Progress()) -> Tuple[str, List[Image.Image]]:
    """
    エージェントの実行フローを制御するメイン関数。
    Gradioのイベントリスナーから呼び出されます。
    """
    logger.info(f"リクエスト受信 - クエリ: '{user_query}', 画像: {input_image is not None}, 動画: {input_video is not None}, 出力言語: {output_lang_display}")
    
    # 選択された表示名からISOコードを取得
    output_language_code = SUPPORTED_LANGUAGES.get(output_lang_display, "en")

    progress(0.1, desc="入力の処理中...")
    context = await InputProcessor.process_input(user_query, input_image, input_video)
    logger.debug(f"入力コンテキスト: {context.keys()}")

    progress(0.3, desc="タスクを分解中...")
    subtasks = await TaskDecomposer.decompose_task(context)
    logger.debug(f"分解されたサブタスク: {len(subtasks)}個")

    progress(0.5, desc="タスクを並列実行中...")
    task_results = await ParallelTaskExecutor.execute_tasks(subtasks, context)
    logger.debug(f"タスク実行結果: {len(task_results)}個")

    progress(0.8, desc="結果を統合中...")
    final_response_text = await ResultSynthesizer.synthesize_results(task_results, context, output_language_code)
    
    # Gradioで表示するためのキーフレーム画像を準備
    display_media_for_gallery = []
    # 画像が入力されていたら、それを表示
    if "image_analysis" in context and context["image_analysis"].get("image_data") and "error" not in context["image_analysis"]:
        display_media_for_gallery.append(context["image_analysis"]["image_data"])
    
    # 動画のキーフレームも表示
    if "video_analysis" in context and context["video_analysis"].get("keyframes") and "error" not in context["video_analysis"]:
        # キーフレームからPIL Imageオブジェクトのみを抽出
        video_keyframes = [kf["image"] for kf in context["video_analysis"]["keyframes"] if "image" in kf]
        display_media_for_gallery.extend(video_keyframes)

    progress(1.0, desc="完了!")
    return final_response_text, display_media_for_gallery

# Gradioインターフェースの定義
with gr.Blocks() as demo:
    gr.Markdown("# マルチモーダルAIエージェント for Hugging Face Spaces")
    gr.Markdown("クエリを入力し、必要に応じて画像または動画をアップロードして分析します。")

    with gr.Row():
        text_input = gr.Textbox(label="あなたのクエリ", placeholder="例：この動画の概要をWeb上の情報も参照してまとめてください。", lines=2)
        image_input = gr.Image(type="pil", label="画像をアップロード（任意）", sources=["upload", "webcam"], height=200)
        video_input = gr.Video(label="動画をアップロード（任意）", height=200)

    output_lang_dropdown = gr.Dropdown(
        label="出力言語を選択",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        value="日本語",
        interactive=True
    )

    submit_button = gr.Button("エージェントを実行")

    output_text = gr.Markdown("## 結果がここに表示されます...")
    output_media = gr.Gallery(label="入力から抽出された画像/キーフレーム", preview=True, height=200)

    submit_button.click(
        fn=run_agent,
        inputs=[text_input, image_input, video_input, output_lang_dropdown],
        outputs=[output_text, output_media],
        api_name="run_agent"
    )

    gr.Examples(
        [
            ["アインシュタインの相対性理論について教えてください。", None, None, "日本語"],
            ["この画像はどこで撮られたものでしょうか？ウェブで調べてください。", os.path.join(os.path.dirname(__file__), "sample_image.jpg") if os.path.exists(os.path.join(os.path.dirname(__file__), "sample_image.jpg")) else None, None, "日本語"],
            ["この動画で何が起こっていますか？関連情報をウェブから検索して詳細を教えてください。", None, os.path.join(os.path.dirname(__file__), "sample_video.mp4") if os.path.exists(os.path.join(os.path.dirname(__file__), "sample_video.mp4")) else None, "English"]
        ],
        inputs=[text_input, image_input, video_input, output_lang_dropdown],
        outputs=[output_text, output_media],
        fn=run_agent, # 例を実行する関数
        cache_examples=False, # デバッグ中はキャッシュを無効にする
    )

# デバッグ用。Hugging Face Spacesでは`app.py`が自動的に起動されます。
# demo.launch()
