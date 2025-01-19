import tensorflow as tf

# Create a SummaryWriter to write events to a log directory
log_dir = "logs/example" 
summary_writer = tf.summary.create_file_writer(log_dir)

# Log scalar values
with summary_writer.as_default():
    for step in range(10):
        loss = 1.0 / (step + 1)  # Example loss value
        tf.summary.scalar('loss', loss, step=step)

# Close the SummaryWriter
summary_writer.close()

print(f"Example tfevents file written to: {log_dir}")