# Project 3: Hurricane Harvey Damage Classification

**Author:** Moksh Nirvaan  
**Model:** Alternate LeNet-5 Convolutional Neural Network  
**Deployment:** Flask-based inference server packaged in a Docker container  
**Docker Hub:** [moksh6/project3](https://hub.docker.com/r/moksh6/project3)  
**Architecture:** Built on x86 (Jetstream class VM)   
**Grader Compliance:** Passed all `/summary` and `/inference` tests 

---

##  Running the Inference Server

This server can be deployed using Docker and Docker Compose.

###  Start the Server

Make sure you're in the `Project3` folder with the `docker-compose.yml` file, then run:

```bash
docker-compose up -d

To stop and remove the running container: 
docker-compose down

Once the server is running on http://localhost:5000, you can interact with it using the following endpoints:

Returns model metadata including name, input/output format, and author.
curl http://localhost:5000/summary

Example response : 
{
  "model": "Alternate LeNet-5",
  "input_size": [3, 128, 128],
  "output_classes": ["damage", "no_damage"],
  "author": "Moksh Nirvaan"
}


POST /inference

Sends an image and receives a damage classification
curl -X POST http://localhost:5000/inference \
  -F "image=@test_image.jpg"

Sample response: 
{ "prediction": "damage" }
