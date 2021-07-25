import json

from paddlenlp.datasets import DatasetBuilder


# SQuAD style dataset
class DuReaderChecklist(DatasetBuilder):
    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            input_data = json.load(f)["data"]

        for entry in input_data:
            title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = []
                    answers = []
                    is_impossible = False

                    if "is_impossible" in qa.keys():
                        is_impossible = qa["is_impossible"]

                    answer_starts = [
                        answer["answer_start"] for answer in qa.get("answers",[])
                    ]
                    answers = [
                        answer["text"].strip() for answer in qa.get("answers",[])
                    ]

                    yield {
                        'id': qas_id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts,
                        'is_impossible': is_impossible
                    }
