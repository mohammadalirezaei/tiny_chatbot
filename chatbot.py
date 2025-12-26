from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/home/mohammad/chatbot/qwen"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded")


class ChatBot:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.system_prompt = (
            "You are a helpful assistant. Remember what the user tells you."
        )

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def clear_history(self):
        self.conversation_history = []
        print("✓ Conversation history cleared!")

    def generate_response(self, user_input, max_new_tokens=256):
        self.conversation_history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_length:]
        assistant_response = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        return assistant_response

def main():
    chatbot = ChatBot(model, tokenizer)
    while True:

        try:
            user_input = input("write:) :").strip()
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("end of chat.. by by ;)0 ")
                break

            if user_input.lower() == "clear":
                chatbot.clear_history()
                continue

            response = chatbot.generate_response(user_input=user_input)
            print(response)
            print()

        except:
            print("something is wrong !!!")


if __name__ == "__main__":
    main()
