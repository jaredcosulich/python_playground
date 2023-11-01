def plot(plt, subplot, y_test, y_pred, title, color):
    plt.subplot(1, 2, subplot)
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color=color)
    plt.title(title)