from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import queue
import threading
import torch
import time

class StreamingAgentSignal:
    """
    input_queue相关
    """
    START_GENERATE = "START_GENERATE"
    CONTINUE_GENERATE = "CONTINUE_GENERATE"
    # 需要在data中提供pos以截断input_ids
    STOP_GENERATE = "STOP_GENERATE"
    # 需要在data中提供text
    APPEND_INPUT = "APPEND_INPUT"
    GENERATION_RESULT = "GENERATION_RESULT"
    def __init__(self, signal_type, data=None):
        self.signal_type = signal_type
        self.data = data if data is not None else {}
        
    def __repr__(self):
        return f"Sinal(type={self.signal_type}, data={self.data})"


class StreamingAgent(threading.Thread):
    def __init__(self, input_queue, output_queue, model, tokenizer, running, decoding, messages=None, debug=False):
        super().__init__()
        # input_queue用于接收外界信号。
        self.input_queue = input_queue
        # output_queue用于生成数据。
        self.output_queue = output_queue
        # 模型相关
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = self.model.config._name_or_path
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.return_legacy_cache = True
        # Used for stream decode and control
        self.token_cache = []
        self.decode_pos = 0
        self.stop_signal = None
        # 运行控制
        self.running = running
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
                                                          add_generation_prompt=True)
                self.decode_pos = len(self.text)
            else:
                raise ValueError("Messages is None for prefilling")
        self.past_key_values = None
        self.input_ids = None
        # 多少个token进行prefill
        self.min_tokens_prefill = 32
        # debug
        self.debug = debug
        self.decoding_input_ids = None

    def run(self):
        while self.running:
            # input_queue优先级最高，包含可能的操作和数据。
            if not self.input_queue.empty():
                self.handle_single(self.input_queue.get())
            
            # decoding
            elif self.decoding:
                if self.debug:
                    print("11111111111")
                if self.decoding_input_ids is None:
                    if self.debug:
                        print("\ndecoding:", self.text)
                        print("----------")
                    self.decoding_input_ids = self.tokenizer.encode(self.text, return_tensors="pt")
                output_ids, self.past_key_values = self._generate(self.decoding_input_ids, self.past_key_values)
                if self.debug:
                    print("2222222222")
                self.input_ids = output_ids
                if self.debug:
                    print("\nself.tokenizer.decode(output_ids[0]):", self.tokenizer.decode(output_ids[0]))
                    print("3333333333")
                self.next_token = output_ids[0][-1]
                if self.next_token == self.tokenizer.eos_token_id:
                    self.decoding = False
                    self.prefilling = True
                    self.put(self.stop_signal)
                else:
                    self.put(self.next_token.tolist())
                self.decoding_input_ids = output_ids
                    
            elif self.prefilling:
                if self.past_key_values is not None:
                    len_past_key_values = self.past_key_values[0][0].size(2)
                else:
                    len_past_key_values = 0
                new_input_ids = self.tokenizer.encode(self.text, return_tensors="pt")
                len_new_input_ids = new_input_ids.size(1)
                # print("len_input_ids, len_past_key_values:", len_new_input_ids, len_past_key_values)
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
                    output_ids, self.past_key_values = self._generate(self.input_ids, self.past_key_values)
                    
    def longest_common_prefix_length(self, tensor1, tensor2):
        # 确定较短的长度
        min_length = min(tensor1.size(0), tensor2.size(0))
        # 比较前 min_length 个元素
        comparison = tensor1[:min_length] == tensor2[:min_length]
        
        # 找到第一个不相同的位置
        mismatch_indices = torch.nonzero(~comparison, as_tuple=False)
        
        if mismatch_indices.numel() == 0:
            # 所有元素都相同
            return min_length
        else:
            # 返回第一个不相同的位置索引
            return mismatch_indices[0].item()

    def cutoff_past_key_values(self, past_key_values, input_cutoff_pos):
        new_past_key_values = []
        for (k, v) in past_key_values:
            sliced_k = k[:, :, :input_cutoff_pos, :] if input_cutoff_pos < k.size(2) else k
            sliced_v = v[:, :, :input_cutoff_pos, :] if input_cutoff_pos < v.size(2) else v
            new_past_key_values.append((sliced_k, sliced_v))
        return tuple(new_past_key_values)
    
    def handle_single(self, signal):
        if signal.signal_type == StreamingAgentSignal.STOP_GENERATE:
            self.output_queue.queue.clear()
            # print("\nSTOP_GENERATE!!!!!!!!!!")
            self.decoding_input_ids = None
            self.decoding = False
            self.prefilling = True
            cutoff_pos = signal.data.get('pos')
            # print("\ncutoff_pos:", cutoff_pos)
            if cutoff_pos:
                self.text = self.text[:cutoff_pos]
                device = next(self.model.parameters()).device
                new_input_ids = self.tokenizer.encode(self.text, return_tensors="pt").to(device)
                reused_length = self.longest_common_prefix_length(new_input_ids[0], self.input_ids[0])
                self.past_key_values = self.cutoff_past_key_values(self.past_key_values, reused_length)
                # print("\nself.text:", self.text)
                # print("\nreused_length:", reused_length)
                self.input_ids = new_input_ids
            len_past_key_values = self.past_key_values[0][0].size(2)
            len_input_ids = self.input_ids.size(1)
            # print("\ncutoff_pos: len_input_ids, len_past_key_values:", len_input_ids, len_past_key_values)

        elif signal.signal_type == StreamingAgentSignal.START_GENERATE:
            self.output_queue.queue.clear()
            self.decoding_input_ids = None
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.START_GENERATE, 
                                                       data={"pos": self.decode_pos}))
            self.decoding = True
            self.prefilling = False
            self.text += """<|im_end|>
<|im_start|>assistant
"""
            self.decode_pos = len(self.text)

        elif signal.signal_type == StreamingAgentSignal.CONTINUE_GENERATE:
            self.output_queue.queue.clear()
            self.token_cache = []
            # print("\nself.token_cache:", self.token_cache)
            # print("\nself.output_queue.qsize():", self.output_queue.qsize())
            self.decoding_input_ids = None
            self.output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.CONTINUE_GENERATE, 
                                                       data={"pos": self.decode_pos}))
            self.decoding = True
            self.prefilling = False
            # self.decode_pos = len(self.text)

        elif signal.signal_type == StreamingAgentSignal.APPEND_INPUT:
            self.decoding_input_ids = None
            assert self.decoding == False, "追加内容不能与decoding同时进行"
            text = signal.data.get('text')
            # print("\nAPPEND_INPUT:", text)
            self.text += text
            self.decode_pos = len(self.text)

    def _generate(self, input_ids, past_key_values):
        """
        目前直接调用model.generate()方法，但是应该重写_sample()的方式，防止重复的预处理 (可能有不会有太大的消耗，maybe?)
        """
        # past_key_values = None
        # attention_mask = torch.ones_like(input_ids)
        
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        if self.debug:
            print(input_ids)
        generate_result = self.model.generate(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=1,
            generation_config=self.generation_config,
            do_sample=False,
        )
        return generate_result.sequences, generate_result.past_key_values
    
    # Simply follow transformers TextStreamer
    # See: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
    def put(self, token):
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
        # print("\ntext:", text)
        # print(last_text + "|" + text + "|")
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
            ):  #
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
            self.decode_pos += len(last_text)
            self.on_finalized_text(last_text, self.decode_pos)
            self.token_cache = [token]

    def on_finalized_text(self, text: str, pos: int):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if text:
            self.text += text
        self.output_queue.put(StreamingAgentSignal(signal_type= StreamingAgentSignal.GENERATION_RESULT, 
                                                   data={"text": text, "pos": pos}))



main_model_name = "/mnt/hwfile/kilab/yinzhenyun/QwQ-32B-Preview/"
main_model_name = "/mnt/hwfile/kilab/yinzhenyun/Qwen2.5-7B-Instruct"


main_model = AutoModelForCausalLM.from_pretrained(
    main_model_name,
    torch_dtype="auto",
    device_map="auto"
)
main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
main_messages = [
    {"role": "system", "content": """有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？"""},
]
main_messages = [
    {"role": "system", "content": """数数，数到200，每数5个数，用两个换行符\n\n分隔。不要解释，直接开始数！
     例子: 1,2,3,4,5\n\n 6,7,8,9,10\n\n 11,12,13,14,15\n\n 16,17,18,19,20\n\n ..."""},
]
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

auxiliary_prompt = """
-任务-
现在我们有一个问题和对应的分析过程，分析过程可能是不完整的。
############
-要求-
你需要按照第一人称"我"的口误，根据解答中的思考过程生成一句简短的话来引导正确的得出答案，注意不要生成超过一句话的内容以避免代替原有分析者的思考。
不要尝试在此输出中去进行计算和具体的推导，当前面的分析偏离正确路径时可以简短的纠正。
当前面的分析按照给出的路径进行时，你只需要肯定并且引导其继续下去。
############
-参考问题-
有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？
############
-参考解答-
1. 设叉子单价为F，勺子单价为S，小刀单价为K。
2. 已知用相同的钱可以买21把叉子和21个勺子，所以总钱数M = 21F + 21S。
3. 同时，用相同的钱也可以买28把小刀，所以 M = 28K。
4. 将两式相等得：21F + 21S = 28K。
把21提出来：21(F + S) = 28K → F + S = (28/21)K = (4/3)K。
5. 一套包含：1把叉子、1个勺子、1把小刀，单价为F + S + K = (4/3)K + K = (7/3)K。
6. 总钱M = 28K，要买整套，x套满足：x * (7/3)K = 28K。
7. 解得x = (28 * 3)/7 = 12套。
8. 因此可以买12套（即12叉、12勺、12刀）用光所有钱。
############
问题：{}
分析(可能不完整):
"""
auxiliary_prompt.format("有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？")
auxiliary_messages = [
    {"role": "system", "content": auxiliary_prompt},
]
auxiliary_messages = [
    {"role": "system", "content": """数数，数到200，每数5个数，用两个换行符\n\n分隔。不要解释，直接开始数！
     例子: 1,2,3,4,5\n\n 6,7,8,9,10\n\n 11,12,13,14,15\n\n 16,17,18,19,20\n\n ..."""},
]
auxiliary_input_queue = queue.Queue()
auxiliary_output_queue = queue.Queue()
auxiliary_agent = StreamingAgent(input_queue = auxiliary_input_queue, output_queue = auxiliary_output_queue, model = auxiliary_model, tokenizer = auxiliary_tokenizer, running=True, decoding=False, messages=auxiliary_messages, debug=False)
auxiliary_agent.start()

aiki_input_queue = queue.Queue()
aiki_output_queue = queue.Queue()

def extract(text):
    """
    处理来自auxiliary模型的输出
    这里是示例实现，你可以根据需求修改处理逻辑
    """
    # 在这里添加你的处理逻辑
    print(f"aiki input: #{text}")
    return text

print("主模型：\n")
last_main_model_generate_len = 0
while True:
    if not main_output_queue.empty():
        signal = main_output_queue.get()
        if signal.signal_type != StreamingAgentSignal.GENERATION_RESULT:
            continue
        text = signal.data.get('text')
        pos = signal.data.get('pos')
        # 主模型生成结束
        if text is None:
            break
        print(text, end = "")
        last_main_model_generate_len += len(text)
        auxiliary_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.APPEND_INPUT, data={"text": text}))
        if "\n\n" in text:
            last_main_model_generate_len = 0
            main_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.STOP_GENERATE, data={"pos": pos}))
            time.sleep(1)
            with main_output_queue.mutex:
                main_output_queue.queue.clear()
            print("\n\n辅助模型：\n")
            auxiliary_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.CONTINUE_GENERATE))
            while True:
                if not auxiliary_output_queue.empty():
                    signal = auxiliary_output_queue.get()
                    if signal.signal_type != StreamingAgentSignal.GENERATION_RESULT:
                        continue
                    text = signal.data.get('text')
                    pos = signal.data.get('pos')
                    
                    # 显示辅助模型的输出
                    if text is not None:
                        print(text, end="")
                        
                    # 将auxiliary输出放入aiki_input_queue
                    aiki_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.GENERATION_RESULT, 
                                                            data={"text": text, "pos": pos}))
                    
                    # 处理文本并放入aiki_output_queue
                    processed_text = extract(text)
                    aiki_output_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.GENERATION_RESULT, 
                                                            data={"text": processed_text, "pos": pos}))
                    
                    # 从aiki_output_queue获取处理后的文本并放入main_input_queue
                    if not aiki_output_queue.empty():
                        aiki_signal = aiki_output_queue.get()
                        processed_text = aiki_signal.data.get('text')
                        main_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.APPEND_INPUT, 
                                                                data={"text": processed_text}))
                        print("aiki output:")
                        print(processed_text, end="")
                    
                    if "\n\n" in text or text is None:
                        auxiliary_input_queue.put(StreamingAgentSignal(signal_type=StreamingAgentSignal.STOP_GENERATE, 
                                                                    data={"pos": pos}))
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