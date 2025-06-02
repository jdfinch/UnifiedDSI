
import transformers as hf
import peft


prompt = '''

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an intelligent and knowledgeable assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

# Key Information Types

## Taxi
* departure_time: the time the taxi should to pick up the Traveler, in HH:MM format.
* arrival_time: the time the taxi should arrive at the drop off location, in HH:MM format.

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
    print(generated)

if __name__ == '__main__':
    main()

    # Output from above prompt
    """
## Hotel
* name: Autumn House
        - The name of the hotel, if the Traveler is looking for a specific hotel.
* location: East
        - The preferred location or area of the hotel, e.g., 'downtown', 'beach', etc.
* price_range: Cheap
        - The preferred price range for the hotel room, e.g., 'budget','mid', 'high'.
* stars: 4
        - The preferred star rating of the hotel, ranging from 1 to 5 stars.
* guests: 2
        - The number of guests the Traveler is booking the hotel for.
* duration: 5
        - The number of nights the Traveler plans to stay at the hotel.

## Place to Visit
* name: Churchill College
        - The specific name of the Place to Visit if the Traveler is looking for a particular location.
* type: college
        - The type of the Place to Visit, such as a museum, park, restaurant, or any other specific type.
* location: West
        - The geographic location of the Place to Visit, e.g., city, country, region, or cross-town.
* information: 
        - Any additional information or notes about the Place to Visit that the Traveler might find useful.

## Taxi
* pickup_location: Stoney S Way
        - the preferred location where the taxi should pick up the Traveler.
* dropoff_location: None
        - the preferred location where the taxi should drop off the Traveler.
* departure_time: 24:45
* <|eot_id|>
    """