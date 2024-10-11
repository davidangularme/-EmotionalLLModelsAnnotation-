import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import Dataset
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import train_test_split

# Charger les modèles de base pré-entraînés à tester
model_names = ["facebook/opt-13b", "bigscience/bloom-7b1", "decapoda-research/llama-7b-hf"]

# Charger le dataset d'instruction tuning AAID élargi et diversifié 
train_dataset = load_dataset("aaid_extended")

# Ajouter un module d'extraction de caractéristiques spécifiques aux émotions
emotion_features = load_emotion_lexicons()
train_dataset = add_emotion_features(train_dataset, emotion_features)

# Diviser en train/validation/test
train_data, eval_data, test_data = train_test_split(train_dataset, test_size=0.2, random_state=42) 
eval_data, test_data = train_test_split(eval_data, test_size=0.5, random_state=42)

# Préparer les données pour l'entraînement
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = [f"{task_prompt}{text}" for task_prompt, text in zip(examples['task_prompt'], examples['text'])]
    targets = examples['target'] 
    model_inputs = tokenizer(inputs, max_length=2048, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=256, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
eval_dataset = eval_data.map(preprocess_function, batched=True, remove_columns=eval_data.column_names)
test_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

# Définir la fonction objective pour l'optimisation des hyperparamètres 
def objective(trial):
    
    # Tester différents modèles de base
    model_name = trial.suggest_categorical("model", model_names)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tester différentes valeurs d'hyperparamètres
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
    mixout = trial.suggest_float("mixout", 0.0, 0.2)
    
    # Configurer l'entraînement
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{model_name}",
        learning_rate=lr,
        num_train_epochs=num_epochs, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        mixout=mixout,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Fine-tuning itératif en ajoutant progressivement des tâches plus complexes  
    for subtask in ["sentiment_classification", "sentiment_regression", "emotion_classification", "emotion_regression"]:
        print(f"Fine-tuning on subtask: {subtask}")
        subtask_train_data = train_dataset.filter(lambda example: example['subtask']==subtask)
        subtask_eval_data = eval_dataset.filter(lambda example: example['subtask']==subtask) 
        trainer.train_dataset = subtask_train_data
        trainer.eval_dataset = subtask_eval_data
        trainer.train()
        
    eval_loss = trainer.evaluate()["eval_loss"] 
    
    return eval_loss

# Optimisation des hyperparamètres avec Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Meilleurs hyperparamètres:", study.best_params)
print("Meilleure perte de validation:", study.best_value)

# Entraîner le meilleur modèle sur l'ensemble des données
best_model_name = study.best_params["model"]
best_model = AutoModelForCausalLM.from_pretrained(best_model_name)

training_args = TrainingArguments(
    output_dir=f"./best_emoLLM_model",
    learning_rate=study.best_params["learning_rate"],
    num_train_epochs=study.best_params["num_epochs"], 
    per_device_train_batch_size=study.best_params["batch_size"],
    per_device_eval_batch_size=study.best_params["batch_size"],
    warmup_ratio=study.best_params["warmup_ratio"],
    weight_decay=study.best_params["weight_decay"], 
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,  
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
)

trainer = Trainer(
    model=best_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Évaluer le modèle EmoLLM final sur l'ensemble de test
test_loss = trainer.evaluate(test_dataset)["eval_loss"]
print(f"Perte sur l'ensemble de test: {test_loss:.3f}")

# Sauvegarder le modèle EmoLLM final
trainer.save_model("./final_emoLLM_model")
