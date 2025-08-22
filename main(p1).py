import heapq
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
import folium
from folium.plugins import AntPath
import webbrowser

#Алгоритм Дейкстры
def dijkstra(graph, start, end, criterion='cost', weights=None):
    if weights is None:
        weights = {'cost': 1, 'time': 1, 'transfers': 1}  # По умолчанию все критерии равны

    queue = [(0, start, [])]
    visited = set()
    while queue:
        (total_cost, node, path) = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return path, total_cost
            for neighbor, attributes in graph[node].items():
                if neighbor not in visited:
                    # Считаем общий вес с учетом предпочтений пользователя
                    weighted_cost = (
                        attributes['cost'] * weights['cost'] +
                        attributes['time'] * weights['time'] +
                        attributes['transfers'] * weights['transfers']
                    )
                    heapq.heappush(queue, (total_cost + weighted_cost, neighbor, path))
    return None, float('inf')

#Данные для ML
data = {
    'route': [
        'Moscow-St Petersburg', 'Moscow-Tver-St Petersburg', 'Moscow-Tver-Novgorod-Vyborg',
        'Moscow-Kazan', 'Moscow-Sochi', 'St Petersburg-Kazan', 'St Petersburg-Sochi',
        'Tver-Kazan', 'Novgorod-Kazan', 'Vyborg-Kazan'
    ],
    'time': [240, 300, 390, 480, 600, 360, 720, 420, 540, 660],
    'cost': [2000, 1500, 1300, 2500, 3000, 2200, 3500, 1800, 2000, 2400],
    'transfers': [0, 1, 2, 1, 0, 1, 2, 1, 2, 1],
    'popularity': [0.8, 0.6, 0.4, 0.7, 0.9, 0.5, 0.8, 0.6, 0.4, 0.7],
    'rating': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]  # 1 — хороший маршрут, 0 — плохой
}

df = pd.DataFrame(data)

#Обучение ML-модели
X = df[['time', 'cost', 'transfers', 'popularity']]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=1)  # Используем 1 соседа
model.fit(X_train, y_train)

# --- Ранжирование маршрутов ---
def rank_routes(routes, model, weights=None):
    if weights is None:
        weights = {'cost': 1, 'time': 1, 'transfers': 1}  # По умолчанию все критерии равны

    ranked_routes = []
    for route in routes:
        # Считаем общий вес с учетом предпочтений пользователя
        weighted_score = (
            route['time'] * weights['time'] +
            route['cost'] * weights['cost'] +
            route['transfers'] * weights['transfers']
        )
        # Собираем признаки для маршрута
        features = pd.DataFrame([[route['time'], route['cost'], route['transfers'], route['popularity']]],
                               columns=['time', 'cost', 'transfers', 'popularity'])
        # Предсказываем рейтинг
        rating = model.predict(features)[0]
        ranked_routes.append({**route, 'rating': rating, 'weighted_score': weighted_score})
    # Сортируем по рейтингу и взвешенному score
    return sorted(ranked_routes, key=lambda x: (x['rating'], x['weighted_score']), reverse=True)

# --- Граф маршрутов ---
graph = {
    'Moscow': {'St Petersburg': {'cost': 2000, 'time': 240, 'transfers': 0}},
    'St Petersburg': {'Vyborg': {'cost': 1000, 'time': 120, 'transfers': 0}},
    'Moscow': {'Tver': {'cost': 500, 'time': 90, 'transfers': 0}},
    'Tver': {'Novgorod': {'cost': 700, 'time': 120, 'transfers': 0}},
    'Novgorod': {'Vyborg': {'cost': 800, 'time': 180, 'transfers': 1}},
    'Vyborg': {}
}

# --- Координаты городов для визуализации ---
city_coordinates = {
    'Moscow': (55.7558, 37.6176),
    'St Petersburg': (59.9343, 30.3351),
    'Tver': (56.8587, 35.9176),
    'Novgorod': (58.5215, 31.2755),
    'Vyborg': (60.7139, 28.7528)
}

# --- Визуализация маршрутов на карте ---
def visualize_route(route):
    map_route = folium.Map(location=[55.7558, 37.6176], zoom_start=5)  # Центр карты на Москве
    locations = [city_coordinates[city] for city in route]
    folium.PolyLine(locations, color="blue", weight=2.5, opacity=1).add_to(map_route)
    for city in route:
        folium.Marker(city_coordinates[city], popup=city).add_to(map_route)
    map_route.save("route_map.html")
    webbrowser.open("route_map.html")  # Открываем карту в браузере

# --- Генерация маршрутов ---
def generate_routes(graph, start, end):
    routes = []
    path, _ = dijkstra(graph, start, end)
    if path:
        route_name = '-'.join(path)
        # Получаем характеристики маршрута
        time = sum(graph[path[i]][path[i + 1]]['time'] for i in range(len(path) - 1))
        cost = sum(graph[path[i]][path[i + 1]]['cost'] for i in range(len(path) - 1))
        transfers = len(path) - 2  # Количество пересадок
        popularity = 0.8  # Пример популярности (можно заменить на реальные данные)
        routes.append({
            'route': route_name,
            'time': time,
            'cost': cost,
            'transfers': transfers,
            'popularity': popularity
        })
    return routes

# --- Функция для обработки ввода пользователя ---
def find_routes():
    start = entry_start.get()
    end = entry_end.get()

    # Валидация ввода
    if start not in graph or end not in graph:
        messagebox.showerror("Ошибка", "Начальная или конечная точка не найдена в графе!")
        return

    try:
        cost_weight = float(entry_cost_weight.get())
        time_weight = float(entry_time_weight.get())
        transfers_weight = float(entry_transfers_weight.get())
        if cost_weight < 0 or time_weight < 0 or transfers_weight < 0:
            raise ValueError("Веса не могут быть отрицательными!")
    except ValueError as e:
        messagebox.showerror("Ошибка", f"Некорректные весовые коэффициенты: {e}")
        return

    weights = {'cost': cost_weight, 'time': time_weight, 'transfers': transfers_weight}

    # Генерация маршрутов
    routes = generate_routes(graph, start, end)
    if routes:
        ranked_routes = rank_routes(routes, model, weights=weights)

        # Вывод результатов
        result_text = "Ранжированные маршруты:\n"
        for route in ranked_routes:
            result_text += (
                f"Маршрут: {route['route']}\n"
                f"Время: {route['time']} мин, Стоимость: {route['cost']} руб, Пересадки: {route['transfers']}\n"
                f"Рейтинг: {route['rating']}, Взвешенный score: {route['weighted_score']}\n\n"
            )
        messagebox.showinfo("Результаты", result_text)

        # Визуализация лучшего маршрута
        best_route = ranked_routes[0]['route'].split('-')
        visualize_route(best_route)
    else:
        messagebox.showerror("Ошибка", "Маршрут не найден!")

# --- Создание интерфейса ---
root = tk.Tk()
root.title("Поиск маршрутов")

# Поля для ввода
tk.Label(root, text="Начальная точка:").grid(row=0, column=0)
entry_start = tk.Entry(root)
entry_start.grid(row=0, column=1)

tk.Label(root, text="Конечная точка:").grid(row=1, column=0)
entry_end = tk.Entry(root)
entry_end.grid(row=1, column=1)

tk.Label(root, text="Вес стоимости (cost):").grid(row=2, column=0)
entry_cost_weight = tk.Entry(root)
entry_cost_weight.grid(row=2, column=1)

tk.Label(root, text="Вес времени (time):").grid(row=3, column=0)
entry_time_weight = tk.Entry(root)
entry_time_weight.grid(row=3, column=1)

tk.Label(root, text="Вес пересадок (transfers):").grid(row=4, column=0)
entry_transfers_weight = tk.Entry(root)
entry_transfers_weight.grid(row=4, column=1)

# Кнопка для поиска маршрутов
button_find = tk.Button(root, text="Найти маршруты", command=find_routes)
button_find.grid(row=5, column=0, columnspan=2)

# Запуск интерфейса
root.mainloop()
