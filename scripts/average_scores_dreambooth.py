import argparse, os, sys, glob

def read_and_average_scores(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    total_image_sim = 0
    total_text_sim = 0
    total_dino_sim = 0
    count = 0

    for line in lines:
        if line.startswith("Image Similarity"):
            # Parse the similarity from the line
            image_sim = float(line.split(":")[1].strip())
            total_image_sim += image_sim
        elif line.startswith("Text Similarity"):
            text_sim = float(line.split(":")[1].strip())
            total_text_sim += text_sim
            count += 1  # Increment count only once per pair of entries
        elif line.startswith("DINO Score"):
            text_sim = float(line.split(":")[1].strip())
            total_text_sim += text_sim
           
        

    # Calculate averages
    if count > 0:
        avg_image_sim = total_image_sim / count
        avg_text_sim = total_text_sim / count
        print(f"Average Image Similarity: {avg_image_sim}")
        print(f"Average Text Similarity: {avg_text_sim}")
        
        with open(file_path, 'a') as file:
            file.write(f"FINAL AVG Image Similarity: {avg_image_sim}\n")
            file.write(f"FINAL AVG Average Text Similarity: {avg_text_sim}\n")
    else:
        print("No data found to compute averages.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tag",
        type=str,
        help="Path to directory with images used to train the embedding vectors"
    )

    opt = parser.parse_args()

    score_file_path = f"all_scores_{opt.tag}.txt" 
    read_and_average_scores(score_file_path)
