import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset

# Đặt đường dẫn tuyệt đối cho import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docchat.embedder import EmbedderFactory
from docchat.store import ChromaVectorStore
from dotenv import load_dotenv

from docchat.llm import LLMConfig, LLMSession

load_dotenv()

# --- Cấu hình Ragas ---
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness
except ImportError:
    print("Vui lòng cài đặt ragas và datasets: uv add --dev ragas datasets")
    sys.exit(1)


def generate_evaluation_dataset(questions: list[dict], data_dir: Path, embedder_provider: str):
    """
    Nhận tập dữ liệu gồm [{"question": "...", "ground_truth": "..."}],
    chạy qua RAG pipeline của DocChat để fill nốt "answer" và "contexts".
    Trả về bộ Dataset chuẩn cho Ragas.
    """
    print(f"🔄 Đang tải hệ thống RAG (Embedder: {embedder_provider}) để generate câu trả lời...")
    embedder = EmbedderFactory.create(embedder_provider)
    store = ChromaVectorStore(embedder=embedder)

    try:
        store.load(data_dir)
    except Exception as e:
        print(f"❌ Không thể load VectorStore tại {data_dir}: {e}")
        return None

    config = LLMConfig()
    results = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print(f"⏳ Bắt đầu điền {len(questions)} câu hỏi kiểm thử...")
    with LLMSession(config) as session:
        for idx, item in enumerate(questions, 1):
            q = item["question"]
            gt = item["ground_truth"]

            # Thực thi quá trình Retrieval để lấy chunks bằng RRF
            k = 5
            search_results = store.search(q, k=k)
            filtered = [r for r in search_results if r.score >= config.min_relevance_score]
            contexts = [r.chunk.text for r in filtered]

            # Fake stream & Generate
            answer = session.complete(q, store, k=k)

            results["question"].append(q)
            results["answer"].append(answer)
            results["contexts"].append(contexts)  # Ragas cần list các str context cho mỗi câu hỏi
            results["ground_truth"].append(gt)

            print(f"  [{idx}/{len(questions)}] Đã xong: {q[:30]}...")

    return Dataset.from_dict(results)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng RAG Pipeline bằng RAGAS.")
    parser.add_argument(
        "--index", default=str(Path.home() / ".docchat"), help="Thư mục chứa ChromaDB index"
    )
    parser.add_argument("--embedder", default="openai")
    parser.add_argument(
        "--dataset",
        default="test_data.json",
        help="File JSON chứa các cặp {question, ground_truth}",
    )
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file {ds_path}.")
        print("💡 Hãy tạo file JSON ví dụ như sau:")
        print(
            '[{"question": "What is Python?", "ground_truth": "Python is a programming language."}]'
        )
        return

    with open(ds_path, encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or len(questions) == 0:
        print("❌ Lỗi: Dataset phải là một Array của JSON objects.")
        return

    hf_dataset = generate_evaluation_dataset(questions, Path(args.index), args.embedder)
    if hf_dataset is None:
        return

    print("\n🚀 Bắt đầu đánh giá tự động bằng Ragas...")
    if "OPENAI_API_KEY" not in os.environ:
        print(
            "⚠️ Cảnh báo: Ragas cần OPENAI_API_KEY để phân tích điểm số. Vui lòng thêm vào `.env`."
        )

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Khởi tạo mô hình Giám khảo (Judge)
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    except ImportError:
        print("Vui lòng đảm bảo đã có langchain-openai (uv add langchain-openai)")
        evaluator_llm = None
        evaluator_embeddings = None

    evaluation_result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\n" + "=" * 50)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ (RAGAS SCORE)")
    print("=" * 50)
    print(evaluation_result)

    # Xuất ra CSV dễ xem
    df = evaluation_result.to_pandas()
    csv_path = "ragas_evaluation_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu file báo cáo chi tiết: {csv_path}")


if __name__ == "__main__":
    main()
