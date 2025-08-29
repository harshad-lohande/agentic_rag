# src/agentic_rag/evaluation/ground_truth.py


def get_ground_truth_data():
    """
    Provides a sample ground truth dataset for RAG evaluation.
    In a real-world scenario, this would be a carefully curated dataset.
    """
    return [
        {
            "question": "What is the difference between MCTs and LCTs?",
            "ground_truth_answer": "The unique structure of MCTs (medium-chain triglycerides) make them extremely beneficial.\n"
            "They are more easily absorbed by the body and can be used as a quick source of energy.\n"
            "Since they don’t require to be processed by digestive enzymes, the body and brain use them almost immediately.\n"
            "LCTs (long-chain triglycerides), on the other hand, have more carbon atoms. They are more complex and take longer to digest.",
        },
        {
            "question": "How do MCTs actually function in burning fat?",
            "ground_truth_answer": "Today there are many studies that show how MCTs can help with weight loss and metabolic health.\n"
            "1. Accelerate Fat Loss and Waist Circumference: One study had participants consume either MCT oil or olive oil \n"
            "to see which fat would help them lose more body fat. Researchers noticed that those consuming MCT \n"
            "oil also reduced their body weight, lowered their total fat mass, and decreased abdominal fat and waist circumference\n"
            "2. Target Abdominal Fat: Another study found that weight loss was “significantly greater in the MCT group.” \n"
            "The MCT group also had a significant reduction in abdominal fat.\n"
            "3. Appetite Control: MCTs may also help with appetite control.\n"
            "4. Help Regulate Blood Sugar: MCTs may also help regulate blood sugar levels, \n"
            "which can be beneficial for those with insulin resistance or type 2 diabetes."
            "5. Prevent Future Weight Gain: Another body of research shows that since MCTs are processed so much quicker than \n"
            "long-chain triglycerides, they have a very low tendency to deposit as body fat.\n"
            "When study participants were given MCTs in one trial, the results conveyed that the \n"
            "subjects increased their ability to burn fat, improved their energy expenditure, \n"
            "and lost more adipose body fat compared to those who didn’t take MCTs — and kept it off.",
        },
        {
            "question": "Other than fat loss, what are the other benefits of MCTs?",
            "ground_truth_answer": "MCTs have several other benefits beyond fat loss, including:\n"
            "1. Improved Cognitive Function: MCTs can provide a quick source of energy for the brain, \n"
            "which may enhance cognitive function and mental clarity.\n"
            "2. Neuroprotective Effects: MCTs may have neuroprotective benefits for wide range of diseases including \n"
            "demetia, Alzheimer's, Parkinson's, stroke, and traumatic brain injury. \n"
            "3. Delayed Brain Aging: MCTs may help delay brain aging by providing extra fuel to repair brain damage.\n"
            "4. Alleviating Depression: MCTs have shown to help alleviate depression due to their antioxidant properties.\n"
            "5. Optimize Intestinal Health: MCTs can help optimize intestinal health by improving gut microbiota.\n"
            "Among other things, your good bacteria produce dozens of neurotransmitters, \n"
            "including more than 90% of your total serotonin and 50% of your dopamine.",
        },
    ]
