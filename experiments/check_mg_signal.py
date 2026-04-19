import matplotlib.pyplot as plt
from data.generators.mackey_glass import MackeyGlassGenerator


def validate_mg():
    gen = MackeyGlassGenerator(length=5000)
    x = gen.generate()

    print("Min:", x.min())
    print("Max:", x.max())
    print("Mean:", x.mean())
    print("Std:", x.std())

    # Plot first 500 points
    plt.figure(figsize=(10, 4))
    plt.plot(x[:500])
    plt.title("Mackey-Glass Signal (First 500 steps)")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    # plt.show()
    plt.savefig('mg_frst_500_steps.png')

    # Plot full signal (compressed)
    plt.figure(figsize=(10, 4))
    plt.plot(x)
    plt.title("Full Mackey-Glass Signal")
    # plt.show()
    plt.savefig('full_signal_mg.jpg')


if __name__ == "__main__":
    validate_mg()