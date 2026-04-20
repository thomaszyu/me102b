import asyncio
import math
import moteus


async def main():
    c1 = moteus.Controller(id=1)
    c2 = moteus.Controller(id=2)
    c3 = moteus.Controller(id=3)
    c4 = moteus.Controller(id=4)

    controllers = [c1, c2, c3, c4]

    for c in controllers:
        await c.set_stop()
    for c in controllers:
        await c.set_recapture_position_velocity()

    while True:
        for c in controllers:
            r = await c.set_position(position=math.nan, query=True)
            print(f"M{r.id}: {r.values[moteus.Register.POSITION]:.3f}", end="  ", flush=True)
        print(flush=True)

        await asyncio.sleep(0.02)


if __name__ == "__main__":
    asyncio.run(main())
