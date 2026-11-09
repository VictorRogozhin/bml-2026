"""Демонстрация табличного Q-learning для мини-лабиринта.

Состояния:
    0 — A (старт)
    1 — B (развилка)
    2 — G (цель)

Действия:
    0 — влево
    1 — вправо

Важно: это табличный Q-learning, а не Deep RL/DQN.
Для запуска требуется Python 3; внешние библиотеки не нужны.
"""

import random

# Для воспроизводимого результата можно раскомментировать следующую строку:
# random.seed(7)

START = 0
BRANCH = 1
GOAL = 2

LEFT = 0
RIGHT = 1

ALPHA = 0.5      # скорость обучения
GAMMA = 0.9      # дисконтирование будущих наград
EPSILON = 0.15   # вероятность случайного исследования
EPISODES = 500
MAX_STEPS = 10

# Q[state][action] — оценка полезности действия в состоянии.
Q = [[0.0, 0.0] for _ in range(3)]


def step(state: int, action: int) -> tuple[int, float]:
    """Выполнить действие и вернуть (новое состояние, награду)."""
    if state == START:
        if action == RIGHT:
            return BRANCH, -1.0
        return START, -2.0

    if state == BRANCH:
        if action == RIGHT:
            return GOAL, 10.0
        return START, -1.0

    # В терминальном состоянии цель остаётся достигнутой.
    return GOAL, 0.0


def choose_action(state: int) -> int:
    """ε-greedy: иногда исследуем случайное действие."""
    if random.random() < EPSILON:
        return random.randrange(2)

    # При равных значениях max выберет первое действие.
    return max(range(2), key=lambda action: Q[state][action])


def train() -> None:
    """Обучить таблицу Q на серии эпизодов."""
    for _ in range(EPISODES):
        state = START

        for _ in range(MAX_STEPS):
            action = choose_action(state)
            next_state, reward = step(state, action)

            # После достижения цели будущая награда равна нулю.
            if next_state == GOAL:
                target = reward
            else:
                target = reward + GAMMA * max(Q[next_state])

            # Табличное обновление Q-learning.
            Q[state][action] += ALPHA * (
                target - Q[state][action]
            )

            state = next_state
            if state == GOAL:
                break


def main() -> None:
    train()

    print("Q(A):", [round(value, 3) for value in Q[START]])
    print("Q(B):", [round(value, 3) for value in Q[BRANCH]])

    best_action = max(range(2), key=lambda action: Q[START][action])
    action_name = "вправо" if best_action == RIGHT else "влево"
    print("Лучшее действие из A:", action_name)


if __name__ == "__main__":
    main()
