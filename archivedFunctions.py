
# Reason for archival: Too Complex
def saveModelScore(model, x, y, path = "kerasModel"):
    # Get the accuracy
    acc = 0
    
    for i in range(preds.shape[0]):
        
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
   
    accuracy = acc / preds.shape[0]

    # Save it to the current path
    path = path + "/scores"

    # Write to the filePath
    file = open(path, "w+")

    # Clear the score then write the score to the file
    file.write(accuracy)

    # Close the file
    file.close()


# Reason for archival: Too Complex
def loadModelScore(path = "kerasModel"):
    # Accuracy is 0 from the get go
    acc = 0

    # Open the .scr file
    with open(path + "/scores", "r") as file:
        # Read the file line by line
        for line in file.readlines():
            Read all the scores in
            scores = [float(i) for i in line.split(",") if i.strip()]
            acc = scores[0]

    # Return the accuracy
    return acc