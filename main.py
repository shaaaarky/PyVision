#!/usr/bin/env/ python3

import sqlite3
import uuid
import cv2 as cv
import face_recognition_models
import face_recognition
import argparse
import numpy as np

def dbConnect(dbFilePath):
	sql_statement = """
	CREATE TABLE IF NOT EXISTS identities (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		encoding BLOB NOT NULL,
		imagePath TEXT NOT NULL,
		creationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)
	"""
	connection = sqlite3.connect(dbFilePath)
	cursor = connection.cursor()

	cursor.execute(sql_statement)
	connection.commit()
	return connection

# Converts from a 128-d numpy array to raw bytes for storage
def serializeEncoding(encoding):
	return encoding.tobytes()

def deserialize_embedding(blob):
    """Turns bytes back into a numpy 128-d array."""
    return np.frombuffer(blob, dtype=np.float64)

# Search for faces in the input image. 
def encodeFromImage(imagePath, model):
	bgr_image = cv.imread(imagePath)
	rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
	boxes = face_recognition.face_locations(rgb_image, model=model)
	print(f"[*] Boxes: {boxes}")
	encodings = face_recognition.face_encodings(rgb_image, boxes)
	print(f"[*] Encodings: {encodings}")
	return encodings 

# Match the to any faces in the db	
def queryDb(encodingToMatch):
	# Pull the whole table off the database 
	connection = dbConnect("identities.db")
	cursor = connection.cursor()
	cursor.execute("SELECT id, encoding FROM identities")
	rows = cursor.fetchall()
	connection.close()

	if not rows:
		print("[!] No identities in the database")
		return False

	known_ids = []
	known_encodings = []
	for id, blob in rows:
		known_ids.append(id) 
		known_encodings.append(deserialize_embedding(blob))

	matches = face_recognition.compare_faces(known_encodings, encodingToMatch)	
	distances = face_recognition.face_distance(known_encodings, encodingToMatch)

	if True in matches:
		best_match_index = np.argmin(distances)  # The smallest distance
		name = known_ids[best_match_index]
		distance = distances[best_match_index]
		print(f"[+] Match found: {name} (Distance: {distance:.4f})")
		return id
	else:
		return False

def createNewDbEntry(encoding, imagePath):
	print("[*] Creating new entry in db ")
	encodingBytes = serializeEncoding(encoding)

	connection = dbConnect("identities.db")
	cursor = connection.cursor()

	try:
		cursor.execute(
		"INSERT INTO identities (encoding, imagePath) VALUES (?, ?)", 
		(encodingBytes, imagePath))
		connection.commit()
		print(f"[*] Succesfully added id: {cursor.lastrowid} to the database")
	except:
		print("[!] Failed to add new user to database")
	connection.close()

def main():
	# Take in image path 
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Image to encode")
	ap.add_argument("-m", "--model", required=False, default="cnn", choices=["cnn", "hog"],help="Model to use. CNN is more accurate but more expensive but hog is cheaper and less accurate")
	args = vars(ap.parse_args())
	imagePath = args["image"]
	model = args["model"]
	encodings = encodeFromImage(imagePath, model)

	# Make sure we found at least one face in the image 
	if not encodings:
		print("[!] No faces found in the image")
		return

	# For each face we find, we query the db 
	for encoding in encodings:
		id = queryDb(encoding)
		if not id:
			print("[!] Face not found in db.")
			createNewDbEntry(encoding, imagePath)
			
if __name__ == '__main__':
	main()