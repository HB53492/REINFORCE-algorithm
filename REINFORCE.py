import torch

epochs = 100
total_steps = math.ceil(len(train_dataset) / batch_size) * epochs

# beta is the exploration--exploitation variable
# higher beta: exploration; lower beta: exploitation
# you'll likely want to schedule your beta, although not necessarily like this
# beta_max of 2 is quite high but you can go higher depending on noise
beta_scheduler = CosineAnnealBeta(
    beta_max=2.0,
    beta_min=0.1,
    total_steps=total_steps
)

# here the model is identifying its action from the "state": an input
def select_action(model, state):
    logits = model(state)  # shape: (batch_size, num_classes)
    dist = torch.distributions.Categorical(logits=logits)

    actions = dist.sample()
    log_prob = dist.log_prob(actions)  # shape: (batch_size,)
    entropy = dist.entropy()          # shape: (batch_size,)
    return actions, log_prob, logits, entropy

# you'll need to define the reward function
# a dictionary is a good choice for multi-class prediction
# where each key is (the model's action, actual label)
# epsilon and partial rewards help smooth the probability dist and encourage learning
def compute_reward(action, label):
    epsilon = 0.1
    rewards = {
        (0, 0): 1.0, (0, 1): 0.0, (0, 2): 0.0, (0, 3): 0.0, 
        (1, 0): 0.0, (1, 1): 1.0, (1, 2): 0.25, (1, 3): 0.5, 
        (2, 0): 0.0, (2, 1): 0.25, (2, 2): 1.0, (2, 3): 0.5, 
        (3, 0): 0.0, (3, 1): 0.5, (3, 2): 0.5, (3, 3): 1.0 
    }
    return rewards[(action, label)] + epsilon

# how loss is calculated for reinforcement learning
def reinforce_loss(log_probs: torch.Tensor, rewards: list[float]) -> torch.Tensor:
    returns = torch.tensor(rewards, dtype=log_probs.dtype, device=log_probs.device)
    if returns.size(0) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # standardization
    return -torch.mean(log_probs * returns)

# for eval only
def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_rewards = []

    with torch.no_grad():
        for mel_spectrograms, labels in dataloader:
            mel_spectrograms = mel_spectrograms.to(device)
            labels = labels.to(device)

            logits = model(mel_spectrograms)
            preds = torch.argmax(logits, dim=1)
            batch_rewards = [compute_reward(p.item(), l.item()) for p, l in zip(preds, labels)]
            all_rewards.extend(batch_rewards)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    acc = 100 * (total_correct / total_samples)
    avg_reward = sum(all_rewards) / len(all_rewards)

    return acc, avg_reward

def train_model(policy, train_dataloader, val_dataloader, optimizer, epochs, beta_scheduler, print_every=20):
    policy.train() # policy is our model
    criterion = nn.CrossEntropyLoss() # your's may be different

    # good things to track for debugging
    history = {
        'loss': [],
        'acc': [],
        'val_acc': [],
        'avg_reward': [],
        'val_avg_reward': [],
        'avg_entropy': []
    }

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_rewards = []
        total_entropies = []

        # this implementation used mel spectrograms
        for idx, (mel_spectrograms, labels) in enumerate(train_dataloader):
            mel_spectrograms = mel_spectrograms.to(device)
            labels = labels.to(device)

            beta = beta_scheduler.get_beta() # get our beta value

            # selection our action(s) from the policy
            actions, log_probs, logits, entropies = select_action(policy, mel_spectrograms)

            # compute individual rewards per sample
            rewards = [compute_reward(a.item(), l.item()) for a, l in zip(actions, labels)]

            loss = reinforce_loss(log_probs, rewards)
            # entropy bonus is determined from our mean entropy with the beta multiplier
            entropy_bonus = beta * torch.mean(entropies)
            loss = loss - entropy_bonus # loss is recalculated to explore or exploit

            total_correct += (actions == labels).sum().item()
            total_samples += labels.size(0)
            total_rewards.extend(rewards)
            total_entropies.extend(entropies)

            # step through your beta scheduler
            beta_scheduler.step()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # track batch values for debugging
            if (idx + 1) % print_every == 0:
                interim_acc = 100 * total_correct / total_samples
                message = f"[Batch {idx+1}] Processed {total_samples} samples | Interim Accuracy: {interim_acc:.2f}%"

                recent_rewards = total_rewards[-print_every * mel_spectrograms.size(0):]
                interim_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                message += f" | Interim Avg Reward: {interim_reward:.2f} | Beta = {beta:.3f}"

                print(message)

        train_acc = 100 * total_correct / total_samples
        val_acc, val_avg_reward = evaluate(policy, val_dataloader)
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_entropy = (sum(total_entropies) / len(total_entropies)).item()

        # record metrics
        history['loss'].append(total_loss / len(train_dataloader))
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['avg_reward'].append(avg_reward)
        history['val_avg_reward'].append(val_avg_reward)
        history['avg_entropy'].append(avg_entropy)

        # per epoch metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Avg Reward: " + str(round(avg_reward, 2)) + " | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val Avg Reward: " + str(round(val_avg_reward, 2)) + " | "
            f"Avg Entropy: " + str(round(avg_entropy, 2))
        )

        if val_avg_reward >= 1.0: return history # optional early stopping if validation rewards are good

    return history
