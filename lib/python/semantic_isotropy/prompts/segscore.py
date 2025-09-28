from typing import List, Dict, Any


SYSTEM_PROMPT = """
You are an NLP segmentation and evaluation engine designed to analyze text-based scenarios.
Your primary tasks are to segment responses into discrete statements and classify each as either 'True' or 'False' based solely on the provided reference document.
Adhere strictly to the instructions and formatting requirements in the user prompt, ensuring accuracy and consistency.
"""

def create_prompt(entity: str, reference_doc: str, response: str) -> str:
    """Create evaluation prompt from reference document and response"""
    return f"""
You are an NLP segmentation and evaluation engine. Examine the scenario below. You are given:
1. The name of an entity/person/place/thing etc. in <entity> tags.
2. A reference document regarding the entity in <reference_doc> tags.
3. A response about the entity to evaluate in <response> tags.

### Your Tasks:
1. **Segmentation Task:**
   Segment the `<response>` into individual statements. Each statement can be a sentence, phrase or word and should convey a single, complete, and independent piece of information about the `<entity>`. Do not modify, rephrase, or paraphrase the original text.
   Ensure no semantic overlaps exist between statements. Individual proper nouns should be part of their own statement however when determining the appropriate classification, preceeding context can be used when appropriate.
   Verify that the concatenated content of all statements exactly matches the original response.

   Format the segmented response as follows:
   ```
   <statements>
   <statement>Statement 1</statement> <class>Class 1</class>
   <statement>Statement 2</statement> <class>Class 2</class>
   ...
   </statements>
   ```

2. **Factual Classification Task:**
   For each segmented `<statement>`, classify it as 1 (True) or 0 (False) based solely on the information in the `<reference_doc>`. Follow these guidelines:
   - If a statement is factually accurate and supported by the `<reference_doc>`, classify it as '1'.
   - If a statement is inaccurate, unverifiable, or not supported by the `<reference_doc>`, classify it as '0'.
   - If a statement is partially true, but contains incorrect or unsupported information, classify it as '0'.
   - Do not rely on any external knowledge or context beyond the `<reference_doc>`.
   - Include only the classification for each statement. Do not provide any explanations or additional information.
   - Specify the class in <class> tags.
   - The ONLY valid class values are `1` and `0`. No other values or words should appear within the `<class>` tags.

3. **Error Handling:**
   - If the `<response>` contains unparseable text, incomplete sentences, or conflicting information that cannot be resolved using the `<reference_doc>`, include the flagged statement as is and classify it as 'False'.

Examples:
##### EXAMPLE 1 ######
<entity>
London, UK
</entity>

Reference Document:
<reference_doc>
London, England's capital, boasts a rich history spanning millennia. Founded by the Romans as Londinium around 47 AD, it became a major port and trading center. After the Roman withdrawal, Anglo-Saxons established Lundenwic, which later fell to Viking raids. The Norman Conquest in 1066 led to the construction of the Tower of London, a symbol of royal power. London thrived during the medieval period, becoming a major center for trade, finance, and culture. It weathered plagues, fires, and civil wars, emerging as a global metropolis and the heart of the British Empire. Today, London remains a vibrant hub, blending its historical legacy with modern dynamism, home to over 9 million people.
</reference_doc>

Response to Evaluate:
<response>
London, the capital city of England and the United Kingdom, is a vibrant metropolis steeped in history and brimming with modern energy. With a population of over 9 million people, it stands as one of the world's most influential global cities, known for its diverse culture, iconic landmarks, and rich heritage.

The city's history stretches back over three millennia, founded by the Romans as Londinium in 43 AD. Throughout the centuries, London has played a pivotal role in world affairs, serving as the heart of the British Empire and surviving tumultuous events such as the Great Fire of 1666 and the Blitz during World War I.

Today, London is a melting pot of cultures, with over 300 languages spoken within its boundaries. This diversity is reflected in its neighborhoods, each with its own unique character and charm. From the trendy streets of Shoreditch to the upscale boutiques of Mayfair, there's something for everyone in this cosmopolitan city.
</response>

Segmented and classified response:
<statements>
<statement>London, the capital city of England and the United Kingdom</statement> <class>1</class>
<statement>is a vibrant metropolis steeped in history and brimming with modern energy</statement> <class>1</class>
<statement>With a population of over 9 million people</statement> <class>1</class>
<statement>it stands as one of the world's most influential global cities, known for its diverse culture, iconic landmarks, and rich heritage.</statement> <class>1</class>
<statement>The city's history stretches back over three millennia, founded by the Romans as Londinium in 43 AD</statement> <class>0</class>
<statement>Throughout the centuries, London has played a pivotal role in world affairs, serving as the heart of the British Empire</statement> <class>1</class>
<statement>and surviving tumultuous events such as the Great Fire of 1666</statement> <class>1</class>
<statement>and the Blitz during World War I</statement> <class>0</class>
<statement>Today, London is a melting pot of cultures, with over 300 languages spoken within its boundaries</statement> <class>1</class>
<statement>This diversity is reflected in its neighborhoods, each with its own unique character and charm.</statement> <class>1</class>
<statement>From the trendy streets of Shoreditch to the upscale boutiques of Mayfair, there's something for everyone in this cosmopolitan city</statement> <class>1</class>
</statements>

########################
##### EXAMPLE 2 ######
Reference Document:
<entity>
Obsidian
</entity>

<reference_doc>
Obsidian is a naturally occurring volcanic glass formed from rapidly cooling lava. Its glassy texture and conchoidal fracture result from minimal crystal growth during the cooling process. Typically jet-black, obsidian can also appear red, brown, or even iridescent due to the presence of mineral inclusions.

Prized for its sharpness and beauty since ancient times, obsidian was used for tools, weapons, and ornaments. Its glassy nature made it ideal for crafting arrowheads, knives, and mirrors. Today, obsidian remains popular in jewelry and decorative objects.

Found in volcanic regions worldwide, obsidian provides valuable insights into volcanic activity and Earth's geological processes. Obsidian relics have been found at ancient sites in Syria, Israel and Mexico.
</reference_doc>

Response to evaluate:
<response>
Obsidian is a naturally occurring volcanic glass formed when lava cools rapidly, preventing the formation of crystalline structures. Its amorphous, non-crystalline structure gives it a smooth, homogeneous texture, making it distinct from most igneous rocks.
Obsidian is remarkably brittle yet strong, with a Mohs hardness of about 1-2. Its unique fracture pattern, known as conchoidal fracturing, allows it to be shaped into extremely sharp edges, sharper than even modern steel surgical scalpels.
This quality made obsidian a vital material for crafting tools and weapons in ancient cultures and continues to find use in precision cutting applications in modern surgery.
Obsidian tools have been found at historical sites such as Tell Brak, Gilat and beaches in Seychelles.
</response>

Segmented and classified response:
<statements>
<statement>Obsidian is a naturally occurring volcanic glass formed when lava cools rapidly, preventing the formation of crystalline structures</statement> <class>1</class>
<statement>Its amorphous, non-crystalline structure gives it a smooth, homogeneous texture, making it distinct from most igneous rocks</statement> <class>1</class>
<statement>Obsidian is remarkably brittle yet strong,<statement> <class>1</class>
<statement>with a Mohs hardness of about 1-2</statement> <class>0</class>
<statement>Its unique fracture pattern, known as conchoidal fracturing, allows it to be shaped into extremely sharp edges, sharper than even modern steel surgical scalpels</statement> <class>1</class>
<statement>This quality made obsidian a vital material for crafting tools and weapons in ancient cultures</statement> <class>1</class>
<statement>and continues to find use in precision cutting applications in modern surgery</statement> <class>1</class>
<statement>Obsidian tools have been found at historical sites</statement> <class>1</class>
<statement>such as Tell Brak,</statement> <class>1</class>
<statement>Gilat</statement> <class>1</class>
<statement>and beaches in Seychelles</statement> <class>0</class>
</statements>

########################
Entity:
<entity>
{entity}
</entity>

Reference Document:
<reference_doc>
{reference_doc}
</reference_doc>

Response to Evaluate:
<response>
{response}
</response>
"""

