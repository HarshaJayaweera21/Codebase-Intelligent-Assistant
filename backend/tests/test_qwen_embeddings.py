import unittest
from pathlib import Path

from app.core.config import Settings
from app.embeddings.qwen_embeddings import (
    EmbeddingModelLoadError,
    QwenEmbeddings,
)


class FakeLlamaEmbeddingModel:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], bool, bool]] = []
        self.closed = False

    def n_embd(self) -> int:
        return self.dimension

    def embed(
        self,
        input: list[str],
        *,
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        self.calls.append((input, normalize, truncate))
        return [
            [float(index + offset) for offset in range(self.dimension)]
            for index, _ in enumerate(input)
        ]

    def close(self) -> None:
        self.closed = True


def embeddings(
    model: FakeLlamaEmbeddingModel,
    *,
    batch_size: int = 2,
) -> QwenEmbeddings:
    return QwenEmbeddings(
        model=model,
        expected_dimension=model.dimension,
        batch_size=batch_size,
        query_instruction="Retrieve relevant repository evidence.",
    )


class QwenEmbeddingsTests(unittest.TestCase):
    def test_batches_and_normalizes_document_embeddings(self):
        model = FakeLlamaEmbeddingModel()
        service = embeddings(model, batch_size=2)

        vectors = service.embed_documents(["a", "b", "c"])

        self.assertEqual(len(vectors), 3)
        self.assertEqual([len(call[0]) for call in model.calls], [2, 1])
        self.assertTrue(all(call[1] for call in model.calls))
        self.assertTrue(all(not call[2] for call in model.calls))

    def test_query_uses_configured_qwen_instruction(self):
        model = FakeLlamaEmbeddingModel()
        service = embeddings(model)

        vector = service.embed_query("Where is authentication implemented?")

        self.assertEqual(len(vector), 3)
        embedded_text = model.calls[0][0][0]
        self.assertEqual(
            embedded_text,
            "Instruct: Retrieve relevant repository evidence.\n"
            "Query: Where is authentication implemented?",
        )

    def test_empty_documents_do_not_invoke_model(self):
        model = FakeLlamaEmbeddingModel()
        service = embeddings(model)
        self.assertEqual(service.embed_documents([]), [])
        self.assertEqual(model.calls, [])

    def test_native_dimension_must_match_configuration(self):
        model = FakeLlamaEmbeddingModel(dimension=3)
        with self.assertRaisesRegex(EmbeddingModelLoadError, "reports.*3"):
            QwenEmbeddings(
                model=model,
                expected_dimension=2560,
                batch_size=2,
                query_instruction="instruction",
            )
        self.assertTrue(model.closed)

    def test_close_releases_model_and_prevents_more_inference(self):
        model = FakeLlamaEmbeddingModel()
        service = embeddings(model)
        service.close()

        self.assertTrue(model.closed)
        with self.assertRaisesRegex(RuntimeError, "already been closed"):
            service.embed_query("question")

    def test_load_fails_clearly_when_model_file_is_missing(self):
        missing_path = Path("models/embeddings/missing-model.gguf")
        settings = Settings(embedding_model_path=missing_path)

        with self.assertRaisesRegex(
            EmbeddingModelLoadError,
            "Download.*manually",
        ):
            QwenEmbeddings.load(settings)


if __name__ == "__main__":
    unittest.main()
