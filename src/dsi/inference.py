
import transformers as hf
import peft


prompt = '''

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an intelligent and knowledgeable assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

# Key Information Types

# Dialogue
User: yes , i am looking for a hotel by the name of the autumn house . can you tell me about it ?
Agent: yes , the autumn house is on the east part of time , the price -s are cheap and it s 4 stars . is there anything else you would like to know ?
User: no , i just want to book it for 2 people for 5 nights starting wednesday .
Agent: you are booked into autumn house . the reference number is e4qftbqt . may i help with anything else ?
User: thank you very much , can you also give some recommendations for place -s to go for college students ?
Agent: churchill college located in the west is a nice place . would you like more info ?
User: just confirming it s in the west part of town ?
Agent: churchill college is located on stoney s way on the west side of town . can i help you with anything else today ?
User: can you also book a taxi for me ?
Agent: sure where are you going to and when ?
User: i am leaving the college to the hotel by 24:45 .
Agent: your taxi has been reserved . it is a white skoda and you can contact it at 07059964954 .

Identify Key Information Values from the Dialogue using the Key Information Types. If there is Key Information that does not fit into any existing Key Information Types, create an appropriate new Information Type for the Value with a description.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# Key Information Values

'''.strip()+'\n\n'


def main():
    adapter_id = 'jdfinch/ssi_dots_lora'

    # Load PEFT config to get the base model
    peft_config = peft.PeftConfig.from_pretrained(adapter_id)
    base_model = peft_config.base_model_name_or_path

    # Load base model
    model = hf.AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="auto")
    tokenizer = hf.AutoTokenizer.from_pretrained(base_model)

    # Load adapter on top
    model = peft.PeftModel.from_pretrained(model, adapter_id)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=False,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print only new generation
    generated = tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated[len(prompt):].strip())

if __name__ == '__main__':
    main()


    """
## Hotel
* name: Autumn House
        - The name of the hotel, if the traveler is looking for a specific hotel.
* location: East
        - The preferred location or area of the hotel.
* price_range: Cheap
        - The preferred price range for the hotel room.
* stars: 4
        - The preferred star rating of the hotel.
* guests: 2
        - The number of guests the traveler is booking the hotel for.
* duration: 5
        - The number of nights the traveler plans to stay at the hotel.

## Travel Plans
* hotel: Autumn House
        - Preferred hotel for the trip, if any.
* meal_option: None
        - Preferred meal option for the travel plans.
* activity_level: None
        - Preferred level of physical activity for the travel plans.
* price_range: None
        - Preferred budget range for the travel plans.
* duration: 5
        - Duration of the trip in days.
* start_date: 2023-07-05
        - Start date of the trip in YYYY-MM-DD format.
* start_time: 24:45
        - Start time of the trip in 24-hour format (HH:MM).
* location: None
        - Preferred location or region for the travel plans.

## Local Insights
* name: Churchill College
        - The name of a specific Local Insight the Traveler is looking for.
* type: college
        - The type of local insight, such as'sightseeing', 'dining','shopping', 'entertainment', 'history', or 'education'.
* area: west
        - The area of interest, such as 'downtown', 'beach', 'city center', etc.
* price_range: None
        - The price range for the Local Insight, such as 'budget','mid', 'high'.
* popularity_score: None
        - The popularity score (out of 10) for the Local Insight, if available.
* description: None
        - A description of the Local Insight, if available.

## Transportation
* type: taxi
        - The type of transportation preferred, such as 'car', 'airplane', 'train', 'bus', or 'taxi'.
* time: 24:45
        - The preferred time for the transportation to be, in 24-hour format 'HH:MM'.
* <|eot_id|>
    """