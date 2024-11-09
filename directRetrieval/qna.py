import json
import itertools
from typing import List, Dict
import jinja2
from .load_qna import load_qna_OOP, QnA_Item
from .llm_utils import generate_response
from .LLMInterfaces import LLMInterface, LlamaCPPServer, AsyncLlamaCPPServer, OpenAI, OpenAISync

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You will be shown a list of questions that {interviewee} answered before (QnA). Your task will be to select the most relevant item from the QnA to answer that question. If the question already exists in the QnA, you should select it. If not, you should select the most relevant question and answer pair that can be used to answer the given question.

The answer should be a JSON object with the following fields:
{json_explanation}

List of questions (QnA):
```{qna}```

Use the following information to understand the question if needed: 
```{info}```"""

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You will be shown a list of questions that {{ interviewee }} answered before (QnA). Your task will be to select the most relevant item from the QnA to answer that question. If the question already exists in the QnA, you should select it. If not, you should select the most relevant question and answer pair that can be used to answer the given question.

The answer should be a JSON object with the following fields:
{{ json_explanation }}

List of questions (QnA):
```{{ qna }}```{% if additionalInformation %}

Use the following information to understand the question if needed:
```{{ additionalInformation }}```
{% endif %}
"""

class QnAModel:
    ANSWER_KEY: str = "Question_and_Answer_From_QnA"
    llm: LLMInterface
    qna: List[QnA_Item]
    systemPromptTemplate: str
    additionalInformation: Dict
    interviewee: str
    interviewer: str
    configPath: str

    def __init__(self, llm: LLMInterface, qna: List[QnA_Item], additionalInformation: Dict, interviewee: str, interviewer: str, systemPromptTemplate: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE):
        self.llm = llm
        self.qna = qna
        self.systemPromptTemplate = systemPromptTemplate
        self.additionalInformation = additionalInformation
        self.interviewee = interviewee
        self.interviewer = interviewer
        self.configPath = ""

    @classmethod
    def fromConfigFile(cls, llm: LLMInterface, configPath: str) -> "QnAModel":
        with open(configPath, "r", encoding='utf-8') as f:
            config = json.load(f)
        if "systemPromptTemplate" not in config:
            return QnAModel(llm, load_qna_OOP(config["qna"]), config["additionalInformation"], config["interviewee"], config["interviewer"])
        else:
            return QnAModel(llm, load_qna_OOP(config["qna"]), config["additionalInformation"], config["interviewee"], config["interviewer"], systemPromptTemplate=open(config["systemPromptTemplate"], 'r', encoding='utf-8').read())
    
    @classmethod
    def fromConfig(cls, llm: LLMInterface, config: Dict) -> "QnAModel":
        if "systemPromptTemplate" not in config:
            return QnAModel(llm, load_qna_OOP(config["qna"]), config["additionalInformation"], config["interviewee"], config["interviewer"])
        else:
            return QnAModel(llm, load_qna_OOP(config["qna"]), config["additionalInformation"], config["interviewee"], config["interviewer"], systemPromptTemplate=open(config["systemPromptTemplate"], 'r', encoding='utf-8').read())

    def getJSONAnswer(self, question: str) -> Dict:
        messages, properties = self.generateQnASelectionPrompt(question=question)
        response = generate_response(self.llm, messages, properties, temperature=0, stream=False)
        assert isinstance(response, dict), "Response is not a dictionary"
        return response

    def getQnA_ID(self, question: str) -> str:
        response = self.getJSONAnswer(question)
        qnaItem: QnA_Item = self.qna[response[self.ANSWER_KEY]["question_number"]]
        return qnaItem.ID
    
    def getAnswer(self, question: str) -> str:
        ID = self.getQnA_ID(question)
        for qnaItem in self.qna:
            if qnaItem.ID == ID:
                return qnaItem.answer
        raise ValueError("ID not found in QnA list")

    def createQnAString(self) -> str:
        questions = self.qna
        newQuestions: List[Dict[str,str | int]] = [
            {
                "tags": " ".join(question.tags),
                "question": question.question,
                "question_number": i,
                "answer": question.answer,
            }
            for i, question in enumerate(questions)]
        return json.dumps(newQuestions, indent=2)

    def createQnAObjectList(self) -> List[Dict]:
        questions = self.qna
        newQuestions: List[Dict[str,str | int]] = [
            {
                "tags": " ".join(question.tags),
                "question": question.question,
                "question_number": i,
                "answer": question.answer,
            }
            for i, question in enumerate(questions)]
        return newQuestions

    # create messages and properties from question, QnA and information
    def generateQnASelectionPrompt(
            self,
            question: str = "",
            # qnaList: List[QnA_Item] = [],
            # information: Dict = {},
            # systemPromptTemplate: str = "",
            # interviewee: str = "",
            # interviewer: str = ""):
            ):
        
        qnaList = self.qna
        information = self.additionalInformation
        systemPromptTemplate = self.systemPromptTemplate
        interviewee = self.interviewee
        interviewer = self.interviewer
        # check if the inputs are valid
        assert question != "", "Question must be provided and not empty"
        assert qnaList != [], "QnA list must be provided and not empty"
        assert systemPromptTemplate != "", "System prompt template must be provided and not empty"
        assert interviewee != "", "Interviewee must be provided and not empty"
        assert interviewer != "", "Interviewer must be provided and not empty"

        qnaString = self.createQnAString()
        dictQNA = self.createQnAObjectList()

        qnaEnums = dictQNA.copy()
        qnaEnums = []
        for qna in dictQNA:
            tags = qna["tags"].split(" ")
            # iterate over all possible orders of the tags
            for perm in itertools.permutations(tags):
                qnaEnums.append({
                    "tags": " ".join(perm),
                    "question": qna["question"],
                    "question_number": qna["question_number"],
                })
        # print(qnaEnums)

        outputs = {
            "Requested_Information":
                {"type": "string",
                "explanation": f"The context of the question and what information the question is aiming to obtain from {interviewee} exactly.",
                },
            "Eventual_answer":
                {"type": "string",
                "explanation": f"Based on the QnA, what do you think {interviewee} would answer to the given question?",
                },
            self.ANSWER_KEY: {
                "type": "object",
                "anyOf": [{"const": qnaEnum} for qnaEnum in qnaEnums],
                "explanation": "Based on the extracted 'Requested_Information', return the verbatim question and answer from the QnA that provides the requested information by the given question, as an object with the fields \"tags\", \"question\", \"question_number\" and \"answer\". Each field MUST be an exact copy of the question and answer from the QnA, not a paraphrase.",
            },
            "Is_answer_in_QnA":
                {"type": "boolean",
                "explanation": "Whether the question was answered in the QnA or not (for example if the selected question and answer pair provides the information to answer the given question).",},
        }


        # extract the explanation of the outputs
        json_explanation = "\n".join([f"{key}: {outputs[key]['explanation']}" for key in outputs])
        # remove the explanation from the outputs
        for key in outputs:
            del outputs[key]["explanation"]
        properties = outputs
        
        info = ""
        for key in information:
            info += f"{key}: {information[key]}\n"
        prompt = f"{interviewer.capitalize()} Question: ```{question}```"
        # systemMessage = systemPromptTemplate.format(json_explanation=json_explanation, info=info, qna=qnaString, interviewee=interviewee, interviewer=interviewer)
        # accept systemPromptTemplate as jinja2 template, in that case use Template.render
        systemMessage = jinja2.Template(systemPromptTemplate).render(json_explanation=json_explanation, additionalInformation=info, qna=qnaString, interviewee=interviewee, interviewer=interviewer)


        messages=[
                {
                    "role": "user",
                    "content": systemMessage,
                },
                {
                    "role": "assistant",
                    "content": "Understood, what's the question?",
                },
                {"role": "user", "content": prompt},
            ]
        # save messages to a file
        with open("messages.txt", "w", encoding='utf-8') as f:
            for message in messages:
                f.write(message["content"] + "\n")
        return messages, properties

    def simplePrompt(self, question: str):
        prompt = f"""You will be shown a list of questions that {self.interviewee} answered before (QnA). Your task will be to select the question_number of the most relevant item from the QnA to answer that question.

IMPORTANT: The answer should be just an integer.

List of Questions (QnA):
```{self.createQnAString()}```
"""
        messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": "Understood, I'll answer the next message with just an integer, what's the question?",
                },
                {"role": "user", "content": question},
            ]
        # save messages to a file
        with open("messages.txt", "w", encoding='utf-8') as f:
            for message in messages:
                f.write(message["content"] + "\n")
        return messages
    
    def simpleAnswer(self, question: str):
        prompt = self.simplePrompt(question)
        response = generate_response(self.llm, prompt, properties=None, temperature=0, stream=False)
        try:
            question_number = int(response)
            qnaItem = self.qna[question_number]
            return qnaItem.answer
        except:
            return ""
    
    def simpleID(self, question: str):
        prompt = self.simplePrompt(question)
        response = generate_response(self.llm, prompt, properties=None, temperature=0, stream=False)
        try:
            question_number = int(response)
            qnaItem = self.qna[question_number]
            return qnaItem.ID
        except:
            print(f"Error: {response}")
            return ""
        
    def evaluateSimple(self, q_a_pairs: List[tuple[str, str]]):
        results = []
        correct = 0
        total = 0
        for question, targetID in q_a_pairs:
            total += 1
            ID = self.simpleID(question)
            if ID == targetID:
                correct += 1
            results.append((question, ID, targetID))
            print(f"Correct: {correct}/{total}")
        return results
    
    def evaluate(self, q_a_pairs: List[tuple[str, str]]):
        results = []
        correct = 0
        total = 0
        for question, targetID in q_a_pairs:
            total += 1
            ID = self.getQnA_ID(question)
            if ID == targetID:
                correct += 1
            results.append((question, ID, targetID))
            print(f"Correct: {correct}/{total}")
        return results
        
if __name__ == "__main__":
    import os
    llm_ = LlamaCPPServer("http://localhost:8080/v1/chat/completions")
    # llm_ = AsyncLlamaCPPServer("http://localhost:8080/v1/chat/completions")
    # llm_ = OpenAISync(os.environ["OPENAI_API_KEY"])
    # qnaModel = QnAModel.fromConfigFile(llm_, "cris.config")

    config = {}
    with open("cris.config", "r", encoding='utf-8') as f:
        config = json.load(f)
    qnaModel = QnAModel.fromConfig(llm_, config)
    # print(qnaModel.getAnswer("What is a freelancer?"))
    # print(qnaModel.simpleAnswer("What is a freelancer?"))

    cris_evaluation = [
    ("Can you tell me a bit about yourself?", "about_me"),
    ("Who is Cristian Desivo?", "about_me"),

    ("What types of projects do you usually work on?", "jobs"),
    ("What are your main areas of expertise?", "jobs"),

    ("Could you explain what mathematical optimization involves?", "mathematical_optimization"),
    ("What is the scope of mathematical optimization work?", "mathematical_optimization"),

    ("What does AI development work typically include?", "ai_development"),
    ("Can you describe what AI development tasks involve?", "ai_development"),

    ("What does freelancing mean for you?", "freelancer"),
    ("How would you describe freelancing?", "freelancer"),

    ("What do you charge for your work?", "rates"),
    ("How much do your services usually cost?", "rates"),

    ("Which programming languages do you specialize in?", "programming_languages"),
    ("What coding languages are you most familiar with?", "programming_languages"),

    ("Which Python frameworks do you commonly use?", "python_framework"),
    ("What are your preferred Python frameworks?", "python_framework")
]

    for result in qnaModel.evaluate(cris_evaluation):
        if result[1] != result[2]:
            print(result)

    # qnaModel_William = QnAModel(llm_, load_qna_OOP("./../william1.qna"), {}, "William", "Investigator")


    # william1_evaluation = [
    #     # ("Who are you?", "name_william"),
    #     ("What's your name?", "name_william"),
    #     ("Can you introduce yourself?", "name_william"),
    #     ("May I ask your name?", "name_william"),

    #     # ("How old are you?", "age_63_william"),
    #     ("What's your age?", "age_63_william"),
    #     ("Can you tell me your age?", "age_63_william"),
    #     ("How many years old are you?", "age_63_william"),

    #     # ("What's your occupation?", "retired_picker"),
    #     ("What do you do for work?", "retired_picker"),
    #     ("Can you tell me your job?", "retired_picker"),
    #     ("Are you employed, and if so, what do you do?", "retired_picker"),

    #     # ("What's a collectibles picker?", "job_description_picker"),
    #     ("Can you explain what a collectibles picker does?", "job_description_picker"),
    #     ("What does it mean to be a collectibles picker?", "job_description_picker"),
    #     ("What kind of work does a collectibles picker do?", "job_description_picker"),

    #     # ("What kind of items do you pick?", "picker_item_variety"),
    #     ("What items do you collect?", "picker_item_variety"),
    #     ("What types of things do you pick up as a picker?", "picker_item_variety"),
    #     ("What sort of collectibles do you usually deal with?", "picker_item_variety"),

    #     # ("Where were you the night of the incident?", "home_on_incident"),
    #     ("Can you tell me where you were the night the incident occurred?", "home_on_incident"),
    #     ("Where were you when the incident happened?", "home_on_incident"),
    #     ("On the night of the incident, where were you?", "home_on_incident"),

    #     # ("What were you doing?", "watching_tv_incident"),
    #     ("What were you up to that night?", "watching_tv_incident"),
    #     ("Can you describe what you were doing at the time of the incident?", "watching_tv_incident"),
    #     ("What activity were you engaged in that night?", "watching_tv_incident"),

    #     # ("Did you hear anything unusual?", "first_gunshot_heard"),
    #     ("Did you hear anything strange that night?", "first_gunshot_heard"),
    #     ("Did you hear any noises during the incident?", "first_gunshot_heard"),
    #     ("Did any sounds catch your attention that night?", "first_gunshot_heard"),

    #     # ("What did you do when you heard the noise?", "investigate_first_shot"),
    #     ("How did you react when you heard the noise?", "investigate_first_shot"),
    #     ("What was your response when you heard the sound?", "investigate_first_shot"),
    #     ("What actions did you take when you heard the noise?", "investigate_first_shot"),

    #     # ("What did you do after hearing the second gunshot?", "called_police_second_shot"),
    #     ("How did you respond after hearing the second gunshot?", "called_police_second_shot"),
    #     ("What actions did you take after the second gunshot?", "called_police_second_shot"),
    #     ("What did you do when you heard the second shot?", "called_police_second_shot"),

    #     # ("Could you have killed the victim?", "alibi_couldnt_kill_john"),
    #     ("Is it possible that you could have killed John?", "alibi_couldnt_kill_john"),
    #     ("Were you in a position to have killed the victim?", "alibi_couldnt_kill_john"),
    #     ("Do you think you could have killed John?", "alibi_couldnt_kill_john"),

    #     # ("What was your relationship with the victim?", "knew_john_slightly"),
    #     ("How did you know the victim?", "knew_john_slightly"),
    #     ("Can you describe your relationship with John?", "knew_john_slightly"),
    #     ("What kind of relationship did you have with the victim?", "knew_john_slightly"),

    #     # ("What did you usually talk about with the victim?", "weather_garden_neighborhood"),
    #     ("What were your typical conversations with John about?", "weather_garden_neighborhood"),
    #     ("What topics did you and John often discuss?", "weather_garden_neighborhood"),
    #     ("When you talked with John, what subjects would come up?", "weather_garden_neighborhood"),

    #     # ("Did you have any conflicts with the victim?", "no_conflict_john"),
    #     ("Were there any disputes between you and John?", "no_conflict_john"),
    #     ("Did you and John ever argue or fight?", "no_conflict_john"),
    #     ("Did you experience any issues with the victim?", "no_conflict_john"),

    #     # ("What's your relationship with Louie?", "relationship_with_louie"),
    #     ("How do you know Louie?", "relationship_with_louie"),
    #     ("Can you describe your relationship with your neighbor Louie?", "relationship_with_louie"),
    #     ("What kind of relationship do you have with Louie?", "relationship_with_louie"),

    #     # ("Did John recently have any conflict with anyone that you know of?", "conflicts_john"),
    #     ("Do you know if John had any recent arguments or fights?", "conflicts_john"),
    #     ("Was John involved in any disputes recently?", "conflicts_john"),
    #     ("Are you aware of any recent conflicts John may have had?", "conflicts_john")
    # ]

    # eval_result = qnaModel_William.evaluateSimple(william1_evaluation)
    # for result in eval_result:
    #     if result[1] != result[2]:
    #         print(result)