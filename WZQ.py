import pygame
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Define constants
WIDTH, HEIGHT = 600, 700
GRID_SIZE = 9  # Define the size of the chessboard, specifically as GRID_SIZE * GRID_SIZE
CELL_SIZE = WIDTH // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create a window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("五子棋")

# Initialize the chessboard
board = np.zeros((GRID_SIZE, GRID_SIZE))
current_player = 1  # 1: Black, 2: White
game_over = False

scores = {1: 0, 2: 0}

# Q-learning parameters
alpha = 0.1
gamma = 0.95
initial_epsilon = 1.0
min_epsilon = 0.3
epsilon_decay = 0.998
Q = {}


# Get status
def get_state(board):
    return tuple(map(tuple, board))

# Check if there are n connected in a line
def check_n_in_a_row(board, player, n):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、对角线
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if board[y][x] == player:
                for dx, dy in directions:
                    count = 1
                    for i in range(1, n):
                        if 0 <= y + i * dy < GRID_SIZE and 0 <= x + i * dx < GRID_SIZE and board[y + i * dy][x + i * dx] == player:
                            count += 1
                        else:
                            break
                    if count >= n:
                        return True
    return False

# Check if you or AI win
def check_win_for_player(board, player):
    return check_n_in_a_row(board, player, 5)

# Check for potential threats (three or four pieces)
def check_potential_threat(board, player, n):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    threats = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if board[y][x] == 0:  # Indicate an empty space
                for dx, dy in directions:
                    count = 0
                    for i in range(1, n + 1):
                        if 0 <= y + i * dy < GRID_SIZE and 0 <= x + i * dx < GRID_SIZE and board[y + i * dy][x + i * dx] == player:
                            count += 1
                        else:
                            break
                    for i in range(1, n + 1):
                        if 0 <= y - i * dy < GRID_SIZE and 0 <= x - i * dx < GRID_SIZE and board[y - i * dy][x - i * dx] == player:
                            count += 1
                        else:
                            break
                    if count >= n - 1:  # Identify potential threats
                        threats.append((y, x))
    return threats

# Choose the action
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0])
    else:
        opponent = 3 - current_player

        # The threat priority of the four chess pieces is the highest
        threats = check_potential_threat(board, opponent, 4)
        if threats:
            return threats[0]

        # Subsequently, it will prevent threats from the three chess pieces
        threats = check_potential_threat(board, opponent, 3)
        if threats:
            return threats[0]

        # If there is no threat, choose the action with the highest value in the Q table
        return max([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0], key=lambda x: Q.get((state, x), 0))

# Update Q-Table
def update_Q(state, action, reward, next_state):
    old_value = Q.get((state, action), 0)
    next_max = max([Q.get((next_state, a), 0) for a in [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0]])
    new_value = old_value + alpha * (reward + gamma * next_max - old_value)
    Q[(state, action)] = new_value

# Calculate reward
def calculate_reward(board, player):
    reward = 0
    opponent = 3 - player  # 对手

    # Check if win
    if check_win_for_player(board, player):
        return 1  # 获胜奖励

    # Check if lose
    if check_win_for_player(board, opponent):
        return -1  # 失败惩罚

    # Check if there have connected four chess pieces together
    if check_n_in_a_row(board, player, 4):
        reward += 0.9

    # Check if the opponent has connected four pieces in a row
    if check_n_in_a_row(board, opponent, 4):
        reward -= 0.8

    # Check if there have connected three chess pieces together
    if check_n_in_a_row(board, player, 3):
        reward += 0.5

    # Check if the opponent has connected three pieces in a row
    if check_n_in_a_row(board, opponent, 3):
        reward -= 0.5

    return reward

# Draw a chessboard
def draw_board():
    screen.fill((200, 150, 100))
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT - 100), 1)
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), 1)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if board[y][x] == 1:
                pygame.draw.circle(screen, BLACK, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 2)
            elif board[y][x] == 2:
                pygame.draw.circle(screen, WHITE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 2)
    # Display score
    font = pygame.font.SysFont(None, 55)
    text = font.render(f"Black(You): {scores[1]}  White(AI): {scores[2]}", True, BLACK)
    screen.blit(text, (10, HEIGHT - 80))

# Training ai
def train_ai(episodes=1000):
    epsilon = initial_epsilon

    total_reward = 0
    learned_states = set()
    rewards = []
    for episode in range(episodes):
        board = np.zeros((GRID_SIZE, GRID_SIZE))
        state = get_state(board)
        while True:
            action = choose_action(state, epsilon)
            row, col = action
            board[row][col] = 1
            reward = calculate_reward(board, 1)
            if check_win_for_player(board, 1):
                update_Q(state, action, reward, get_state(board))
                break
            else:
                next_state = get_state(board)
                update_Q(state, action, reward, next_state)
                state = next_state
            # AI's opponent randomly play chess
            available_moves = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0]
            if not available_moves:
                break
            opponent_move = random.choice(available_moves)
            board[opponent_move[0]][opponent_move[1]] = 2
            if check_win_for_player(board, 2):
                reward = -1
                update_Q(state, action, reward, get_state(board))
                break
            state = get_state(board)
        # Attenuation exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        # Calculate the average reward and Q-table coverage
        total_reward += reward
        learned_states.add((state, action))
        rewards.append(reward)
    print("Average Reward:", total_reward / episodes)
    print("Q-Table Coverage:", len(learned_states) / (3 ** (GRID_SIZE * GRID_SIZE)))
    # Draw charts to represent the reward curve and Q-value distribution
    plt.figure()
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.show()
    plt.figure()
    q_values = [Q[key] for key in Q if key[1] is not None]
    plt.hist(q_values, bins=50)
    plt.xlabel('Q Value')
    plt.ylabel('Frequency')
    plt.title('Q-Value Distribution')
    plt.show()

# Saving Q-Table
def save_q_table(filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q, f)

# Loading Q-Table
def load_q_table(filename="q_table.pkl"):
    global Q
    with open(filename, "rb") as f:
        Q = pickle.load(f)

# Main loop
def main():
    global current_player, game_over, board, scores
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_over:
                    # If the game ends and the right mouse button is clicked, reset the chessboard
                    if event.button == 3:
                        board = np.zeros((GRID_SIZE, GRID_SIZE))  # Clear the chessboard
                        game_over = False  # Reset game status
                        current_player = 1  # Reset the current player
                else:
                    # Left click processing before the game ends
                    if event.button == 1:
                        x, y = event.pos
                        col = x // CELL_SIZE
                        row = y // CELL_SIZE
                        if board[row][col] == 0:
                            board[row][col] = current_player
                            if check_win_for_player(board, current_player):
                                scores[current_player] += 1
                                game_over = True
                            current_player = 3 - current_player  # Switch players

        # AI's turn
        if current_player == 2 and not game_over:
            state = get_state(board)
            action = choose_action(state, min_epsilon)
            row, col = action
            board[row][col] = 2
            if check_win_for_player(board, 2):
                scores[2] += 1
                game_over = True
            current_player = 3 - current_player  # Switch player

        draw_board()
        if game_over:
            font = pygame.font.SysFont(None, 75)
            if 3 - current_player == 1:
                text = font.render(f"You win!", True, RED)
            else:
                text = font.render(f"You lose! Haha", True, RED)
            screen.blit(text, (WIDTH // 2 - 150, HEIGHT // 2 - 50))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    # Training AI
    print("Start training AI...")
    train_ai(episodes=20000)
    save_q_table()
    print("Training completed!")

    # Start the main loop of the game
    main()