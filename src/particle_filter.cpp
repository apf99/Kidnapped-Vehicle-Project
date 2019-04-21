/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;

// Define the randon number generator here so we can use it throughout
default_random_engine gen;

const float EPSILON = 1.0e-5f;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   *   Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   *   Add random Gaussian noise to each particle.
   */
  num_particles = 200;  // TODO: Set the number of particles

  // Create a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> normal_x(x, std[0]);
  normal_distribution<double> normal_y(y, std[1]);
  normal_distribution<double> normal_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle newParticle;
    newParticle.id = i;
    newParticle.x = normal_x(gen);
    newParticle.y = normal_y(gen);
    newParticle.theta = normal_theta(gen);
    newParticle.weight = 1.0f;
    
    particles.push_back(newParticle);
  	weights.push_back(newParticle.weight);  
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */
  
  // Create a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> normal_x(0.0, std_pos[0]);
  normal_distribution<double> normal_y(0.0, std_pos[1]);
  normal_distribution<double> normal_theta(0.0, std_pos[2]);

   for (int i = 0; i < num_particles; i++) {
     double theta = particles[i].theta;
     
     if (fabs(yaw_rate) > EPSILON) {
       particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
       particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
       particles[i].theta += yaw_rate * delta_t;
     } else {
       particles[i].x += velocity*delta_t * cos(theta);
       particles[i].y += velocity*delta_t * sin(theta);
     }
     
     // Add sensor noise
     particles[i].x += normal_x(gen);
     particles[i].y += normal_y(gen);
     particles[i].theta += normal_theta(gen);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */
  for(unsigned int i=0; i < observations.size(); i++) { 
    double minDistance = numeric_limits<double>::max();  // set to maximum to begin
    int minId;
    
    for(unsigned int j=0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(distance < minDistance) {
        minDistance = distance;
        minId = predicted[j].id;
      }
    }
    
    observations[i].id = minId;
  }
}
       
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
   /**
   *   Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   */
  
  // for each particle
  for (int i = 0; i < num_particles; i++) {
    Particle part = particles[i];
    
    // create a list of observations in map coordinates 
    vector<LandmarkObs> map_observations;
    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = observations[j];
      LandmarkObs newLandmark;
      newLandmark.id = obs.id;
      newLandmark.x = part.x + cos(part.theta)*obs.x - sin(part.theta)*obs.y;
      newLandmark.y = part.y + cos(part.theta)*obs.y + sin(part.theta)*obs.x;
      map_observations.push_back(newLandmark);
    }
     
    // create a list of landmarks that are within sensor's range
    vector<LandmarkObs> valid_landmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      LandmarkObs newLandmark;
      newLandmark.id = map_landmarks.landmark_list[j].id_i;
      newLandmark.x = map_landmarks.landmark_list[j].x_f;
      newLandmark.y = map_landmarks.landmark_list[j].y_f;

      if (dist(part.x, part.y, newLandmark.x, newLandmark.y) <= sensor_range) {
        valid_landmarks.push_back(newLandmark);  
      }
    }
  
    // associate the map_observations with the valid_landmarks
    dataAssociation(valid_landmarks,  map_observations);
    
    particles[i].weight = 1.0;     // reset the particle's weight to 1 and recalculate
    // for each observation
    for (unsigned int j = 0; j < map_observations.size(); j++) {
      LandmarkObs obs = map_observations[j];
      
      // find the landmark with the associated id
      LandmarkObs lnd;
      for (unsigned int k = 0; k < valid_landmarks.size(); k++) {
        if (valid_landmarks[k].id == obs.id) {
          lnd = valid_landmarks[k];
          break;
        }
      }
      // calculate the weigh/probability of this particle using the multivariate Gaussian
      double Sx = std_landmark[0];
      double Sy = std_landmark[1];
      double exponent = pow((obs.x - lnd.x), 2) / (2.0*Sx*Sx)  +  pow((obs.y - lnd.y), 2) / (2.0*Sy*Sy);   
      particles[i].weight *= exp(-exponent) / (2.0 * M_PI * Sx * Sy);
    }
    weights[i] = particles[i].weight;
  }  
}

void ParticleFilter::resample() {
  /**
   *   Resample particles with replacement with probability proportional 
   *   to their weight. 
   */
  
   // Vector for new particles
  vector<Particle> resampled_particles (num_particles);
  
  // Use discrete distribution to return particles by weight
  for (int i = 0; i < num_particles; i++) {
    discrete_distribution<int> index(weights.begin(), weights.end());
    resampled_particles[i] = particles[index(gen)];
  }
  
  // Replace old particles with the resampled particles
  particles = resampled_particles;
}
       
void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}