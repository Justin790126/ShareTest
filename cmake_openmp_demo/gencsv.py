import csv
import random

# Function to generate a CSV with specified number of rows and random double values
def generate_csv(filename, num_rows):
    # Define the column names
    fieldnames = ["PD", "PG", "MIN_DIS", "af", "bf", "cf"]

    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        csv_writer.writeheader()

        # Generate random double values for each row and write to the CSV
        for i in range(num_rows):
            # Random double values between 0 and 100
            # pd_value = round(random.uniform(0, 100), 2)  # Random value for "PD"
            # pg_value = round(random.uniform(0, 100), 2)  # Random value for "PG"
            # min_dis_value = round(random.uniform(0, 100), 2)  # Random value for "MIN_DIS"
            pd_value = i
            pg_value = 2*i + 1
            min_dis_value = 2*i + 2
            a_value = round(random.uniform(0, 100), 2)
            b_value = round(random.uniform(0, 100), 2)
            c_value = round(random.uniform(0, 100), 2)
            # Create a dictionary representing the row
            row = {
                "PD": pd_value,
                "PG": pg_value,
                "MIN_DIS": min_dis_value,
                "af": a_value,
                "bf": b_value,
                "cf": c_value
            }

            # Write the row to the CSV
            csv_writer.writerow(row)


# Get the number of rows from user input
num_rows = 300000

# Generate the CSV with the given number of rows
generate_csv("data.csv", num_rows)
