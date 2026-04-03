from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

log_file = sys.argv[1]
event_acc = EventAccumulator(log_file)
event_acc.Reload()
for tag in event_acc.Tags()['scalars']:
    if 'val' in tag or 'loss' in tag:
        events = event_acc.Scalars(tag)
        print(f"Tag: {tag}, Latest Step: {events[-1].step}, Value: {events[-1].value:.4f}")
