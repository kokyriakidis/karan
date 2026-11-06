"""Random cases for the ksw_extd2 oracle: one line each, flag w zdrop end_bonus q e q2 e2 a b ambi target query."""
import random
import sys

random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)


def mutate(s, rate):
    out = []
    for c in s:
        x = random.random()
        if x < rate / 3:
            out.append(random.choice("ACGT"))
        elif x < 2 * rate / 3:
            pass
        elif x < rate:
            out.append(c)
            out.append(random.choice("ACGT" * 5 + "N"))
        else:
            out.append(c)
    return "".join(out)


def seq(n):
    s = []
    while len(s) < n:
        r = random.random()
        if r < 0.15:
            s += random.choice("ACGT") * random.randint(2, 9)
        elif r < 0.25:
            u = "".join(random.choice("ACGT") for _ in range(random.randint(2, 5)))
            s += u * random.randint(2, 6)
        else:
            s.append(random.choice("ACGT"))
    return "".join(s[:n])


LENGTHS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,5,17,40,100,250,600").split(",")]
scores = [(4, 2, 24, 1, 2, 4, 1), (39, 3, 81, 1, 1, 19, 1), (6, 2, 26, 1, 2, 4, 1), (4, 2, 24, 1, 2, 4, 0)]
flags = [0, 0x02, 0x40, 0x40 | 0x02 | 0x80, 0x08, 0x08 | 0x10, 0x01, 0x40 | 0x08]
for _ in range(int(sys.argv[1])):
    n = random.choice(LENGTHS)
    t = seq(n)
    q = mutate(t, random.choice([0.0, 0.02, 0.08, 0.2, 0.5]))
    if random.random() < 0.2:
        q = q + seq(random.randint(0, 80))
    if random.random() < 0.1:
        t = t + seq(random.randint(0, 80))
    if not q:
        q = "A"
    gq, ge, gq2, ge2, a, b, ambi = random.choice(scores)
    print(random.choice(flags), random.choice([-1, -1, 3, 10, 40, 150]), random.choice([-1, 20, 100, 200, 400]),
          random.choice([-1, 0, 10, 50]), gq, ge, gq2, ge2, a, b, ambi, t, q)
