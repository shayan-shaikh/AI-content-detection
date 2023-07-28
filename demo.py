from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("./AI-Content-Detector-V2", local_files_only =True)
model = AutoModelForSequenceClassification.from_pretrained("./AI-Content-Detector-V2", local_files_only = True, num_labels=2)
def predict(query):
    tokens = tokenizer.encode(query)
    tokens = tokens[:tokenizer.model_max_length - 2]
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.sigmoid()
    fake, real = probs.detach().cpu().flatten().numpy().tolist() 
    return real
txt = "Once upon a time in a small, picturesque village nestled amidst rolling hills, there lived a young girl named Amelia. Amelia was known for her adventurous spirit and unwavering curiosity. She spent her days exploring the enchanting forests surrounding the village, seeking hidden treasures and uncovering the mysteries of nature. One sunny morning, as Amelia ventured deeper into the woods than ever before, she stumbled upon an old, weathered book lying beneath a majestic oak tree. Intrigued by its ancient appearance, she carefully opened it to reveal delicate pages filled with faded ink. To her astonishment, the book was a portal to a magical world. As Amelia read the words aloud, the surrounding trees rustled and swayed, and a shimmering portal materialized before her eyes. Without hesitation, she stepped through, embarking on the adventure of a lifetime. In the realm beyond the portal, Amelia discovered a land brimming with mythical creatures and captivating landscapes. She befriended a mischievous but lovable gnome named Jasper, who became her trusty companion on this extraordinary journey. Together, Amelia and Jasper encountered talking animals, crossed treacherous bridges, and solved riddles posed by ancient guardians. Each step brought them closer to the heart of the land, where a powerful sorceress resided. Legends whispered that she held the key to granting wishes and restoring balance to their world. Their path was filled with challenges, but Amelia's courage and Jasper's quick wit always prevailed. They faced daunting trials, tested their limits, and discovered the true meaning of friendship and perseverance. Finally, after a perilous climb up a towering mountain, they reached the sorceress's hidden sanctuary. The sorceress, wise and all-knowing, listened to their earnest plea and recognized the pureness in their hearts. In a grand display of magic, the sorceress granted Amelia and Jasper one wish. With unwavering determination, they chose to bring peace and harmony to their village and the world beyond. The sorceress smiled, acknowledging their noble intentions, and the wish was granted. Returning home, Amelia and Jasper were hailed as heroes. The village blossomed with newfound serenity, and tales of their epic adventure spread far and wide. Amelia, forever changed by her journey, continued to explore, inspire, and ignite the spark of adventure in the hearts of others. And so, the tale of Amelia and Jasper became a legend, a reminder that within every person lies the power to embark on extraordinary journeys and make a difference in the world, fueled by courage, curiosity, and the belief that anything is possible."
predictions = predict(txt)
print('Real probability : ',predictions)
if predictions < 0.80:
    print(f"This text is AI generated 🚨 with probability : "+ str(1 - predictions) )
else:
    print(f"This text is real ✅ with a probability : " + str(predictions) )