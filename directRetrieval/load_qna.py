import os
from typing import List

class QnA_Item:
    ID: str
    tags: List[str]
    question: str
    answer: str
    def __init__(self, ID: str, tags: List[str], question: str, answer: str):
        self.ID = ID
        self.tags = tags
        self.question = question
        self.answer = answer

    @staticmethod
    def from_dict(d: dict) -> "QnA_Item":
        return QnA_Item(d["ID"], d["tags"], d["question"], d["answer"])
    
    @staticmethod
    def from_raw(raw: str) -> "QnA_Item":
        qna_lines = raw.split("\n")
        question = qna_lines[2]
        ID = qna_lines[1]
        tags = qna_lines[0].split(" ")
        answers = qna_lines[3:]
        return QnA_Item(ID, tags, question, answers[0])

    def copy(self) -> "QnA_Item":
        return QnA_Item(self.ID, self.tags, self.question, self.answer)


def load_qna(filename):
    # Load the QnA pairs from the file
    qna = []
    with open(filename, 'r') as file:
        text = file.read()
        qnas_raw = text.split("\n\n")
        for i, qna_raw in enumerate(qnas_raw):
            qna_lines = qna_raw.split("\n")
            question = qna_lines[2]
            ID = qna_lines[1]
            tags = qna_lines[0]
            answers = qna_lines[3:]
            qna.append({
                "tags": tags,
                "question": question,
                # "answers": answers, TODO: Add support for multiple answers
                "answer": answers[0],
                "question_number": i+1,
                "ID": ID
            })
    return qna

def load_qna_OOP(filename: str):
    qna: List[QnA_Item] = []
    with open(filename, 'r') as file:
        text = file.read()
        qnas_raw = text.split("\n\n")
        for i, qna_raw in enumerate(qnas_raw):
            qna.append(QnA_Item.from_raw(qna_raw))
    return qna

if __name__ == "__main__":
    filename = "william1.qna"
    test_qna = load_qna(filename)
    # Print the QnA pairs
    for i, qna_pair in enumerate(test_qna):
        print(f"QnA Pair {i+1}")
        print(f"Question: {qna_pair['question']}")
        print(f"Tags: {qna_pair['tags']}")
        print(f"Answer: {qna_pair['answer']}")
        print()

    import json
    print(json.dumps(test_qna, indent=2))