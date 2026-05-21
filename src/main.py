import io
import pstats
import cProfile
import argparse
import traceback
import tracemalloc
from datetime import datetime

from scripts.instances import Gym

if __name__ == "__main__":
    gym_env = None
    try:
        print("Starting earth-gym (open-source / SPICE edition)…")

        start_time = datetime.now()

        parser = argparse.ArgumentParser(description="Earth-Gym RL environment server.")
        parser.add_argument("--host", default="localhost", type=str,  help="Host address.")
        parser.add_argument("--port", default=5555,        type=int,  help="Port number.")
        parser.add_argument("--conf", type=str,  help="Agent configuration JSON file.")
        parser.add_argument("--evpt", type=str,  help="Event-zones CSV file.")
        parser.add_argument("--out",  type=str,  help="Output folder for plots / logs.")
        parser.add_argument("--pro",  type=int,  help="Enable profiling (1) and memory tracing.")
        args = parser.parse_args()

        if args.pro:
            print("Tracking profile and memory allocation…")
            tracemalloc.start()

        gym_env = Gym(args=args)

        if args.pro:
            pr = cProfile.Profile()
            pr.enable()
            gym_env.start(host=args.host, port=args.port)
            pr.disable()

            s  = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
            ps.print_stats(40)
            print(s.getvalue())
            pr.dump_stats("src/main-profile.prof")
        else:
            gym_env.start(host=args.host, port=args.port)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        if args.pro:
            snapshot  = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")
            print("Top 100 memory-consuming lines:")
            for stat in top_stats[:100]:
                print(stat)
            tracemalloc.stop()

        end_time = datetime.now()

        if gym_env is not None and not gym_env.is_shutdown():
            gym_env.shutdown()

        print(f"Time elapsed: {datetime.now() - start_time}")
        print("Earth Gym was shut down. Bye!")
