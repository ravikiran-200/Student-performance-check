"""Generate the fictional Indian-style student dataset used by this project."""

import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "indian_student_performance_demo.csv"
ROWS = 600
SUBJECTS = {
    "applied_data_science_marks": 1.0,
    "machine_learning_marks": -2.0,
    "data_warehouse_and_data_mining_marks": -1.0,
    "coding_skills_marks": 2.0,
    "principles_of_big_data_management_marks": -1.5,
}
FEMALE_NAMES = [
    "Aadhya", "Ananya", "Aarohi", "Diya", "Isha", "Kavya", "Meera", "Nisha",
    "Pooja", "Riya", "Saanvi", "Tara", "Vanya", "Avni", "Bhavna", "Charu",
    "Deepa", "Esha", "Gauri", "Hema", "Jaya", "Kiran", "Lavanya", "Mira",
    "Neha", "Pari", "Radhika", "Sakshi", "Tanvi", "Uma",
]
MALE_NAMES = [
    "Aarav", "Aditya", "Arjun", "Dev", "Ishaan", "Kabir", "Karan", "Krish",
    "Manav", "Neel", "Pranav", "Rahul", "Rohan", "Samar", "Vihaan", "Aman",
    "Aniket", "Darshan", "Harish", "Jay", "Kunal", "Mihir", "Nikhil", "Om",
    "Raj", "Sahil", "Tejas", "Varun", "Yash", "Zubin",
]
SURNAMES = [
    "Sharma", "Patel", "Reddy", "Iyer", "Mehta", "Nair", "Kapoor", "Rao",
    "Kumar", "Desai", "Verma", "Joshi", "Banerjee", "Das", "Singh", "Shah",
    "Menon", "Kulkarni", "Pillai", "Mishra", "Chopra", "Gupta", "Bose", "Sinha",
    "Malhotra", "Naidu", "Bhat", "Saxena", "Ghosh", "Chatterjee", "Jain", "Kaur",
    "Shetty", "Thakur", "Agarwal", "Dutta", "Pawar", "Khan", "Mukherjee", "Roy",
]
FIELDS = [
    "student_id", "student_name", "gender", "semester", "attendance_percent",
    "weekly_study_hours", "assignment_completion_percent", "lab_completion_percent",
    "class_participation_score", *SUBJECTS, "overall_average",
]


def bounded(value, low, high):
    return max(low, min(high, value))


def main():
    random.seed(20261110)
    name_counts = {"Female": 0, "Male": 0}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for number in range(1, ROWS + 1):
            gender = random.choice(["Female", "Male"])
            index = name_counts[gender]
            first_names = FEMALE_NAMES if gender == "Female" else MALE_NAMES
            first_name = first_names[index % len(first_names)]
            surname = SURNAMES[(index // len(first_names)) % len(SURNAMES)]
            name_counts[gender] += 1

            attendance = round(bounded(random.gauss(78, 14), 40, 100))
            study_hours = round(bounded(random.gauss(11.5, 5), 1, 30))
            assignments = round(
                bounded(28 + 0.43 * attendance + 0.55 * study_hours + random.gauss(0, 11), 20, 100)
            )
            labs = round(
                bounded(20 + 0.42 * attendance + 0.90 * study_hours + random.gauss(0, 14), 15, 100)
            )
            participation = random.randint(1, 5)
            base = (
                58
                + 0.20 * (attendance - 75)
                + 1.0 * (study_hours - 10)
                + 0.08 * (assignments - 70)
                + 0.08 * (labs - 65)
                + 1.5 * (participation - 3)
                + random.gauss(0, 10)
            )
            marks = {
                subject: round(bounded(base + subject_effect + random.gauss(0, 7), 0, 100), 1)
                for subject, subject_effect in SUBJECTS.items()
            }
            row = {
                "student_id": f"STU{number:04d}",
                "student_name": f"{first_name} {surname}",
                "gender": gender,
                "semester": random.randint(1, 6),
                "attendance_percent": attendance,
                "weekly_study_hours": study_hours,
                "assignment_completion_percent": assignments,
                "lab_completion_percent": labs,
                "class_participation_score": participation,
                **marks,
                "overall_average": round(sum(marks.values()) / len(marks), 1),
            }
            writer.writerow(row)
    print(f"Created {ROWS} fictional student records: {OUTPUT}")


if __name__ == "__main__":
    main()
