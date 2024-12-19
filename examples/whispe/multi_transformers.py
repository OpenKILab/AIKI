from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import queue
import threading
import torch
import time
import chainlit as cl

class StreamingAgentSignal:
    START_GENERATE = "START_GENERATE"
    CONTINUE_GENERATE = "CONTINUE_GENERATE"
    STOP_GENERATE = "STOP_GENERATE"
    APPEND_INPUT = "APPEND_INPUT"
    GENERATION_RESULT = "GENERATION_RESULT"
    GENERATION_END = "GENERATION_END"
    COMPUTE_LOGPROBS = "COMPUTE_LOGPROBS"

    def __init__(self, signal_type, data=None):
        self.signal_type = signal_type
        self.data = data if data is not None else {}
        
    def __repr__(self):
        return f"Sinal(type={self.signal_type}, data={self.data})"


class StreamingAgent(threading.Thread):
    def __init__(self, input_queue, output_queue, model, tokenizer, running, decoding, messages=None, debug=False):
        super().__init__()
        # 输入输出通信队列
        self.input_queue = input_queue
        self.output_queue = output_queue
        # LLM模型
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = self.model.config._name_or_path
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.return_legacy_cache = True
        # decode
        self.token_cache = []
        self.decode_pos = 0
        # running control
        self.running = running
        self.stop_signal = None
        if decoding:
            self.prefilling = False
            self.decoding = True
            if messages is not None:
                self.text = self.tokenizer.apply_chat_template(messages,
                                                          tokenize=False,
                                                          add_generation_prompt=True)
                self.decode_pos = len(self.text)
            else:
                raise ValueError("Messages is None for decoding")
        else:
            self.prefilling = True
            self.decoding = False
            if messages is not None:
                self.text = self.tokenizer.apply_chat_template(messages,
                                                          tokenize=False,
                                                          continue_final_message=True)
                self.decode_pos = len(self.text)
            else:
                raise ValueError("Messages is None for prefilling")
        # input_ids and past_key_values
        # input_ids 用于保存与 past_key_values 一致的输入
        self.input_ids = None
        self.past_key_values = None
        # for prefilling
        self.min_tokens_prefill = 32
        # for decoding
        # 我们可以从self.text来获得input_ids
        # 但是由于逐个token解码，text不一定与token_ids一致
        # 保留decoding_input_ids用于decoding
        self.decoding_input_ids = None
        # debug
        self.debug = debug

    def longest_common_prefix_length(self, tensor1, tensor2):
        """
        获取两个1D张量的最长公共前缀
        """
        min_length = min(tensor1.size(0), tensor2.size(0))
        comparison = tensor1[:min_length] == tensor2[:min_length]
        mismatch_indices = torch.nonzero(~comparison, as_tuple=False)
        if mismatch_indices.numel() == 0:
            return min_length
        else:
            return mismatch_indices[0].item()

    def cutoff_past_key_values(self, past_key_values, cutoff_pos):
        new_past_key_values = []
        for (k, v) in past_key_values:
            sliced_k = k[:, :, :cutoff_pos, :] if cutoff_pos < k.size(2) else k
            sliced_v = v[:, :, :cutoff_pos, :] if cutoff_pos < v.size(2) else v
            new_past_key_values.append((sliced_k, sliced_v))
        return tuple(new_past_key_values)

    def get_reused_kvcache(self, input_ids, past_key_values, new_input_ids):
        reused_length = self.longest_common_prefix_length(input_ids[0], new_input_ids[0])
        input_ids = input_ids[:, :reused_length]
        past_key_values = self.cutoff_past_key_values(past_key_values, reused_length)
        return input_ids, past_key_values

    def run(self):
        while self.running:
            # input_queue优先级最高，包含可能的操作和数据。
            if not self.input_queue.empty():
                self.handle_single(self.input_queue.get())

            # decoding
            elif self.decoding:
                if self.debug:
                    print("\n[DEBUG]", "run -> self.decoding")
                    print("\n[DEBUG]", "run -> self.decoding -> self.text:", self.text)
                if self.decoding_input_ids is None:
                    print("\n[DEBUG]", "run -> self.decoding -> self.decoding_input_ids is None")
                    self.decoding_input_ids = self.tokenizer.encode(self.text, return_tensors="pt")
                    if self.past_key_values:
                        self.input_ids, self.past_key_values = self.get_reused_kvcache(self.input_ids, self.past_key_values, self.decoding_input_ids)
                # print("=============decoding_input_ids.size(1) s==============")
                # print(self.model_name)
                # print(self.decoding_input_ids.size(1))
                # print("=============decoding_input_ids.size(1) e==============")
                self.input_ids = self.decoding_input_ids
                output_ids, self.past_key_values = self.generate(self.input_ids, self.past_key_values)
                self.decoding_input_ids = output_ids
                next_token = output_ids[0][-1]
                if next_token == self.tokenizer.eos_token_id:
                    self.decoding = False
                    self.prefilling = True
                    self.decode(self.stop_signal)
                else:
                    self.decode(next_token.tolist())
                    
            elif self.prefilling:
                new_input_ids = self.tokenizer.encode(self.text, return_tensors="pt")
                if self.past_key_values:
                    # len_input_ids = self.input_ids.size(1)
                    # print("[TEMP-DEBUG]", "len_input_ids:", len_input_ids)
                    # len_past_key_values = self.past_key_values[0][0].size(2)
                    # print("[TEMP-DEBUG]", "len_past_key_values:", len_past_key_values)
                    self.input_ids, self.past_key_values = self.get_reused_kvcache(self.input_ids, self.past_key_values, new_input_ids)
        
                len_new_input_ids = new_input_ids.size(1)
                if self.past_key_values is not None:
                    len_past_key_values = self.past_key_values[0][0].size(2)
                else:
                    len_past_key_values = 0

                if self.debug:
                    print("[DEBUG]", "len_input_ids, len_past_key_values:", len_new_input_ids, len_past_key_values)

                if len_past_key_values == len_new_input_ids:
                    time.sleep(0.01)
                    continue
                elif len_past_key_values > len_new_input_ids:
                    raise ValueError("KVcache can't be longer than input_ids, something maybe wrong.")
                elif len_new_input_ids - len_past_key_values < self.min_tokens_prefill:
                    time.sleep(0.01)
                    continue
                else:
                    self.input_ids = new_input_ids
                    output_ids, self.past_key_values = self.generate(self.input_ids, self.past_key_values)
    
    def handle_single(self, signal: StreamingAgentSignal):
        if signal.signal_type == StreamingAgentSignal.STOP_GENERATE:
            self.decoding_input_ids = None
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.STOP_GENERATE, 
                                                       data={"pos": self.decode_pos}))
            print("\n[INFO]", "handle_single(STOP_GENERATE)")
            self.token_cache = []
            self.output_queue.queue.clear()
            self.decoding = False
            self.prefilling = True
            stop_pos = signal.data.get('pos')
            if stop_pos:
                self.text = self.text[:stop_pos]
                new_input_ids = self.tokenizer.encode(self.text, return_tensors="pt")
                self.input_ids, self.past_key_values = self.get_reused_kvcache(self.input_ids, self.past_key_values, new_input_ids)

            self.decode_pos = len(self.text)

        elif signal.signal_type == StreamingAgentSignal.START_GENERATE:
            print("=======辅助模型上下文开始==========")
            print("辅助模型： StreamingAgentSignal.START_GENERATE:")
            print(self.text)
            print("=========辅助模型上下文结束========")
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.START_GENERATE, 
                                                       data={"pos": self.decode_pos}))
            self.decoding = True
            self.prefilling = False
            self.text += """<|im_end|>
<|im_start|>assistant
"""
            self.decode_pos = len(self.text)

        elif signal.signal_type == StreamingAgentSignal.CONTINUE_GENERATE:
            print("========主模型上下文开始：=========")
            print("主模型： StreamingAgentSignal.CONTINUE_GENERATE:")
            print(self.text)
            print("=========主模型上下文结束：========")
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.CONTINUE_GENERATE, 
                                                       data={"pos": self.decode_pos}))
            self.decoding = True
            self.prefilling = False

        elif signal.signal_type == StreamingAgentSignal.APPEND_INPUT:
            assert self.decoding == False, "追加内容不能与decoding同时进行"
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.APPEND_INPUT, 
                                                       data={"pos": self.decode_pos}))
            text = signal.data.get('text')
            self.text += text
            self.decode_pos = len(self.text)
        
        elif signal.signal_type == StreamingAgentSignal.COMPUTE_LOGPROBS:
            assert self.decoding == False, "COMPUTE_LOGPROBS不能与decoding同时进行"
            self.decoding_input_ids = self.tokenizer.encode(self.text, return_tensors="pt")
            if self.past_key_values:
                self.input_ids, self.past_key_values = self.get_reused_kvcache(self.input_ids, self.past_key_values, self.decoding_input_ids)
            self.input_ids = self.decoding_input_ids
            output_ids, self.past_key_values = self.generate(self.input_ids, self.past_key_values)
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.COMPUTE_LOGPROBS, 
                                                       data={"logprobs": self.compute_logprobs(self.input_ids, self.past_key_values)}))
            
    def compute_logprobs(self, input_ids, past_key_values):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, past_key_values=past_key_values)
            logits = outputs.logits  
            logprobs = torch.log_softmax(logits, dim=-1)  
        return logprobs


    def generate(self, input_ids, past_key_values):
        """
        目前直接调用model.generate()方法，但是应该重写_sample()的方式，防止重复的预处理 (可能有不会有太大的消耗，maybe?)
        """
        # past_key_values = None
        # attention_mask = torch.ones_like(input_ids)
        if self.debug:
            print("[DEBUG]", "self.generate -> input_ids:", input_ids)
        generate_result = self.model.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            max_new_tokens=1,
            generation_config=self.generation_config,
            do_sample=False,
        )
        return generate_result.sequences, generate_result.past_key_values
    
    # Simply follow transformers TextStreamer
    # See: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
    def decode(self, token):
        last_text = self.tokenizer.decode(self.token_cache)
        if token == self.stop_signal:
            if last_text:
                self.decode_pos += len(last_text)
                self.on_finalized_text(last_text, self.decode_pos)
            self.token_cache = []
            self.on_finalized_text(None, self.decode_pos)
            return
        self.token_cache.append(token)
        text = self.tokenizer.decode(self.token_cache)

        def _is_chinese_char(cp):
            """Checks whether CP is the codepoint of a CJK character."""
            # This defines a "chinese character" as anything in the CJK Unicode block:
            #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
            #
            # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
            # despite its name. The modern Korean Hangul alphabet is a different block,
            # as is Japanese Hiragana and Katakana. Those alphabets are used to write
            # space-separated words, so they are not treated specially and handled
            # like the all of the other languages.
            if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
            ):
                return True
            return False

        if text.endswith("\n"):
            self.decode_pos += len(text)
            self.on_finalized_text(text, self.decode_pos)
            self.token_cache = []

        elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
            self.decode_pos += len(text)
            self.on_finalized_text(text, self.decode_pos)
            self.token_cache = []

        elif text.endswith(" "):
            self.decode_pos += len(text)
            self.on_finalized_text(text, self.decode_pos)
            self.token_cache = []

        # 暂时假设不存在 (word_a_prefix) (word_a_suffix blank word_b_prefix) (word_b_suffix) 这种情况
        elif " " in text.strip():
            if last_text:
                self.decode_pos += len(last_text)
                self.on_finalized_text(last_text, self.decode_pos)
            self.token_cache = [token]

    def on_finalized_text(self, text: str, pos: int):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if text:
            self.text += text
        self.output_queue.put(StreamingAgentSignal(signal_type= StreamingAgentSignal.GENERATION_RESULT, 
                                                    data={"text": text, "pos": pos}))


    def compute_logprobs(self, model, input_ids, attention_mask):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs


color_dic = {
    "main_model": "#FFFFFF",
    "au_model": "#9C9C9C",
    "rag": "#FFFFE0",
}

# ak = AIKI(db_path="/mnt/hwfile/kilab/leishanzhe/db/law_industrycorpus2/")
from xmlrpc.client import ServerProxy

client = ServerProxy('http://localhost:10055')


# main_model_name = "/mnt/hwfile/kilab/yinzhenyun/QwQ-32B-Preview/"
main_model_name = "/mnt/hwfile/kilab/yinzhenyun/Qwen2.5-7B-Instruct"


main_model = AutoModelForCausalLM.from_pretrained(
    main_model_name,
    torch_dtype="auto",
    device_map="auto"
)
main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)

@cl.on_message
async def main(message: cl.Message):
    input_text = message.content.strip()
    """
    甲公司与乙公司签订房屋租赁合同，甲公司是出租方，乙公司是承租方。甲公司在合同上加盖公司印章，并有法定代表人签字，乙公司未加盖公章，仅有法定代表人丙的签字。后因乙公司未支付租金，甲公司将乙公司和丙诉至法院，乙公司答辩称丙未经公司授权签订租赁合同，乙公司未加盖公章，不应承担付款责任。丙答辩称其是职务行为，个人不应承担付款责任，该案应如何认定责任主体？
    请对以下案件做出分析并给出可能的判决：2016年11月29日，原被告双方签订了《中通巨龙电动汽车试销协议》，协议约定原告授权被告为江苏南通市的销售代理商。2016年12月4日、2016年12月16日原告先后向被告发送车辆共计12台。被告应向原告支付货款327200元，但被告收到车辆后，迟迟未支付货款。
    王女士在某公司工作已满四年，在怀孕五个月时告知公司人事并申请公司配合办理生育保险申请等事项，孕期工作表现均符合公司公示制度要求。两周后，公司借口以王女士无法胜任工作为由要与其解除劳动关系，王女士如何维护自己的合法权益？
    请对以下案件做出分析并给出可能的判决：2021年10月13日2时许，被告人尹某持Ｃ1型驾驶证，酒后（经鉴定，尹某血液中乙醇含量为146.51ｍｇ／100ｍｌ）驾驶一辆无号牌二轮摩托车行驶至东莞市××街镇××道高速桥底路段时，被执勤交警查获。
    """
    problem = input_text
    main_prompt = f"""
    你是一个专业人士。请基于问题和检索到的相关信息，进行分析并给出专业的解答。
    在回答时：
    1. 首先分析问题的关键点
    2. 结合检索到的相关信息
    3. 用简短的话复述检索到的相关信息
    3. 给出具体的分析与解答
    
    问题：
    {problem}

    请基于以上问题进行分析。如果后续收到标记为<RAG>的补充信息，请将其作为重要参考依据，复述相关信息后，继续完善你的分析。
    """
    main_messages = [
        {"role": "system", "content": main_prompt},
    ]
    # main_messages = [
    #     {"role": "system", "content": """有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？"""},
    # ]
    # main_messages = [
    #     {"role": "system", "content": """数数，数到200，每数5个数，用两个换行符\n\n分隔。不要解释，直接开始数！
    #      例子: 1,2,3,4,5\n\n 6,7,8,9,10\n\n 11,12,13,14,15\n\n 16,17,18,19,20\n\n ..."""},
    # ]
    main_input_queue = queue.Queue()
    main_output_queue = queue.Queue()
    main_agent = StreamingAgent(input_queue = main_input_queue, output_queue = main_output_queue, model = main_model, tokenizer = main_tokenizer, running=True, decoding=True, messages=main_messages, debug=False)
    main_agent.start()

    auxiliary_model_name = "/mnt/hwfile/kilab/yinzhenyun/Qwen2.5-7B-Instruct"
    auxiliary_model = AutoModelForCausalLM.from_pretrained(
        auxiliary_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(auxiliary_model_name)

    # auxiliary_prompt = """
    # -任务-
    # 现在我们有一个问题和对应的分析过程，分析过程可能是不完整的。
    # ############
    # -要求-
    # 你需要按照第一人称"我"的口误，根据解答中的思考过程生成一句简短的话来引导正确的得出答案，注意不要生成超过一句话的内容以避免代替原有分析者的思考。
    # 不要尝试在此输出中去进行计算和具体的推导，当前面的分析偏离正确路径时可以简短的纠正。
    # 当前面的分析按照给出的路径进行时，你只需要肯定并且引导其继续下去。
    # ############
    # -参考问题-
    # 有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？
    # ############
    # -参考解答-
    # 1. 设叉子单价为F，勺子单价为S，小刀单价为K。
    # 2. 已知用相同的钱可以买21把叉子和21个勺子，所以总钱数M = 21F + 21S。
    # 3. 同时，用相同的钱也可以买28把小刀，所以 M = 28K。
    # 4. 将两式相等得：21F + 21S = 28K。
    # 把21提出来：21(F + S) = 28K → F + S = (28/21)K = (4/3)K。
    # 5. 一套包含：1把叉子、1个勺子、1把小刀，单价为F + S + K = (4/3)K + K = (7/3)K。
    # 6. 总钱M = 28K，要买整套，x套满足：x * (7/3)K = 28K。
    # 7. 解得x = (28 * 3)/7 = 12套。
    # 8. 因此可以买12套（即12叉、12勺、12刀）用光所有钱。
    # ############
    # 问题：{problem}
    # 分析(可能不完整):
    # """
    # auxiliary_prompt.format("有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？")
    # auxiliary_messages = [
    #     {"role": "system", "content": auxiliary_prompt},
    # ]
    # auxiliary_messages = [
    #     {"role": "system", "content": """数数，数到200，每数5个数，用两个换行符\n\n分隔。不要解释，直接开始数！
    #      例子: 1,2,3,4,5\n\n 6,7,8,9,10\n\n 11,12,13,14,15\n\n 16,17,18,19,20\n\n ..."""},
    # ]
    # auxiliary_prompt = auxiliary_prompt.format(problem = problem)
    # auxiliary_messages = [
    #     {"role": "system", "content": auxiliary_prompt},
    # ]

    # query_prompt = """
    # -task-
    # Please summarize the content as a query.
    # -demand-
    # This summarization will be used as a query to search with Bing search engine.
    # The query should be short but need to be specific to promise Bing can find related knowledge or pages.
    # You can also use search syntax to make the query short and clear enough for the search engine to find relevant language data.
    # Try to make the query as relevant as possible to the last few sentences in the content.
    # **IMPORTANT**
    # Just output the query directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
    # You must use the Chinese.
    # DO NOT answer the query.
    # """
    # auxiliary_system_prompt = f"##Instruction: {query_prompt}; \n##Content: \n"

    # auxiliary_system_prompt = """"""
    auxiliary_system_prompt = """
    [Instruction]
    Based on the input step, design a query that can effectively search relevant information with Bing search engine to assist another model in solving a problem. The query should accurately reflect the key information and question points in the input step, ensuring that the retrieved information is useful for the other model to continue thinking and solving the problem.
    [Requirements]
    1. Accuracy: The query must accurately reflect the key information and question points in the input step. 
    2. Relevance: The query should retrieve information relevant to the input step. 
    3. Conciseness: The query should be as concise as possible, avoiding unnecessary information. 
    4. Searchability: The query should be effectively searchable, ensuring that the auxiliary model can retrieve relevant information from the knowledge base. 
    5. Grammar and Spelling: The query should have correct grammar and spelling, ensuring the accuracy of the search results. 
    6. Domain Knowledge: The query should take into account the domain knowledge and professional terminology involved in the input step, ensuring that the retrieved information can be effectively understood and utilized. 
    7. Output Format: You only need to output the query in Chinese.
    [Input step]
    """

    auxiliary_messages = [
        {"role": "system", "content": auxiliary_system_prompt},
    ]

    auxiliary_input_queue = queue.Queue()
    auxiliary_output_queue = queue.Queue()
    auxiliary_agent = StreamingAgent(input_queue = auxiliary_input_queue, output_queue = auxiliary_output_queue, model = auxiliary_model, tokenizer = auxiliary_tokenizer, running=True, decoding=False, messages=auxiliary_messages, debug=False)
    auxiliary_agent.start()

    def extract(text):
        """
        处理来自auxiliary模型的输出
        """
        print(f"\n\naiki input: \n{text}")
        retrieve_data = client.retrieve(text, 1)
        result = ""
        for item in retrieve_data:
            result += item
        return f"""<RAG>{result}<RAG>\n\n"""

    print("主模型：\n")
    last_main_model_generate_len = 0
    msg = cl.Message("")
    await msg.send()
    main_text = ""
    while True:
        if not main_output_queue.empty():
            signal = main_output_queue.get()
            if signal.signal_type != StreamingAgentSignal.GENERATION_RESULT:
                continue
            text = signal.data.get('text')
            pos = signal.data.get('pos')
            if text is None:
                break
            main_text += text
            # 主模型生成结束
            if text is None:
                break
            colored_text = f'<span style="color: {color_dic["main_model"]};">{text}</span>' # 主模型
            await msg.stream_token(colored_text)
            print(text, end = "")
            last_main_model_generate_len += len(text)
            auxiliary_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.APPEND_INPUT, data={"text": text}))
            if "\n\n" in text and last_main_model_generate_len > 100:
                # print("=================================")
                # print(last_main_model_generate_len)
                print("=======主模型生成开始==========")
                print(main_text)
                main_text = ""
                print("=======主模型生成结束==========")
                cutoff_main_model_step_generate_len = last_main_model_generate_len
                last_main_model_generate_len = 0
                main_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.STOP_GENERATE, data={"pos": pos}))
                time.sleep(1)
                with main_output_queue.mutex:
                    main_output_queue.queue.clear()
                print("\n\n辅助模型：\n")
                auxiliary_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.START_GENERATE))
                auxiliary_output = ""  # 用于收集完整的辅助模型输出
                start_generate_pos = -1
                while True:
                    if not auxiliary_output_queue.empty():
                        signal = auxiliary_output_queue.get()
                        if signal.signal_type == StreamingAgentSignal.START_GENERATE:
                                start_generate_pos = signal.data.get('pos')
                                continue
                        if signal.signal_type != StreamingAgentSignal.GENERATION_RESULT:
                            continue
                        text = signal.data.get('text')
                        pos = signal.data.get('pos')
                        if text:
                            colored_text = f'<span style="color: {color_dic["au_model"]};">{text}</span>'  # 辅助模型
                            await msg.stream_token(colored_text)
                        # 显示辅助模型的输出
                        if text is not None:
                            print(text, end="")
                            auxiliary_output += text  # 收集输出
                        
                        if text is None or "\n\n" in text:
                            print("=======辅助模型生成开始==========")
                            print(auxiliary_output)
                            print("=======辅助模型生成结束==========")
                            processed_text = extract(auxiliary_output)
                            colored_text = f'<span style="color: {color_dic["rag"]};">{processed_text}</span>'  # rag
                            await msg.stream_token("\n")
                            await msg.stream_token(colored_text)
                            await msg.stream_token("\n\n")
                            main_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.APPEND_INPUT, 
                                                                    data={"text": processed_text}))
                            print("\n\nAIKI处理结果：")
                            print(processed_text, end="")
                            print("\nAIKI处理结束")
                            
                            auxiliary_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.STOP_GENERATE, 
                                                                        data={"pos": start_generate_pos - cutoff_main_model_step_generate_len}))
                            with auxiliary_output_queue.mutex:
                                auxiliary_output_queue.queue.clear()
                            main_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.CONTINUE_GENERATE))
                            print("\n\n主模型：\n")
                            break
                    else:
                        time.sleep(0.01)
        else:
            time.sleep(0.01)

    print("END")