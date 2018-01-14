#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <GL/glut.h>
#include <iostream>
#include <cmath>

using namespace std;

typedef struct {
	double x, y, z;
} vertex;

class Triangle {
public:
	vector<vertex> v_list;

	Triangle(vertex v1, vertex v2, vertex v3){
		v_list.push_back(v1);
		v_list.push_back(v2);
		v_list.push_back(v3);
	}
};

void EventHandler(sf::RenderWindow *w);
void drawFun(vector<Triangle> fig);
vector<Triangle> genIcosahedron();
vector<Triangle> divideTriangle(Triangle t);
void mesh(vector<Triangle> & fig);
void project(vector<Triangle> & fig);

bool mouse_state = false;
int mouse_x, mouse_y;
int angle_x, angle_z;
int last_angle_x, last_angle_z;
float zoom = 1;

vector<Triangle> ico = genIcosahedron();

int main(int argc, char** argsv){
	sf::RenderWindow w(sf::VideoMode(480,480), "Icosphere");
	w.setFramerateLimit(60);

	/* OpenGl settings */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glEnable(GL_DEPTH_TEST);
	/* !OpenGl settings */

	while(w.isOpen()){
		EventHandler(&w);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		glClearColor(0,0,0,0);

		drawFun(ico);
		w.display();
	}
}

void EventHandler(sf::RenderWindow *w){
	sf::Event e;
	while(w->pollEvent(e)){
		if(e.type == sf::Event::Closed){
			w->close();
		}
		else if (e.type == sf::Event::KeyPressed){
			if(e.key.code == sf::Keyboard::Z){
				mesh(ico);
				project(ico);
			}
		}
		else if (e.type == sf::Event::MouseButtonPressed) {
			mouse_state = true; // setting draggin flag
			// saving current starting point in global vars
			mouse_x = sf::Mouse::getPosition().x;
			mouse_y = sf::Mouse::getPosition().y;
		}
		else if (e.type == sf::Event::MouseButtonReleased){
			mouse_state = false; // clearing draggin flag
			// saving the angles where the object finished after rotation
			last_angle_x = last_angle_x + angle_x;
			last_angle_z = last_angle_z + angle_z;
			// resetting angles
			angle_x = 0;
			angle_z = 0;
		}
		else if (e.type == sf::Event::MouseMoved){
			if(mouse_state){
				// while dragging we get the rotation angles
				angle_x = sf::Mouse::getPosition().y - mouse_y;
				angle_z = sf::Mouse::getPosition().x - mouse_x;
			}
		}
	}
}

vector<Triangle> genIcosahedron(){
	vector<Triangle> ico;
	double aur = (1 + sqrt(5))/2;

	// Purple rectangle
	vertex v1_1;
		v1_1.x = 0;
		v1_1.y = -3;
		v1_1.z = 3*aur;

	vertex v1_2;
		v1_2.x = 0;
		v1_2.y = 3;
		v1_2.z = 3*aur;

	vertex v1_3;
		v1_3.x = 0;
		v1_3.y = 3;
		v1_3.z = -3*aur;

	vertex v1_4;
		v1_4.x = 0;
		v1_4.y = -3;
		v1_4.z = -3*aur;

	// Green rectangle
	vertex v2_1;
		v2_1.x = -3*aur;
		v2_1.y = 0;
		v2_1.z = 3;

	vertex v2_2;
		v2_2.x = 3*aur;
		v2_2.y = 0;
		v2_2.z = 3;

	vertex v2_3;
		v2_3.x = 3*aur;
		v2_3.y = 0;
		v2_3.z = -3;

	vertex v2_4;
		v2_4.x = -3*aur;
		v2_4.y = 0;
		v2_4.z = -3;

	// Light Green rectangle
	vertex v3_1;
		v3_1.x = -3;
		v3_1.y = -3*aur;
		v3_1.z = 0;

	vertex v3_2;
		v3_2.x = -3;
		v3_2.y = 3*aur;
		v3_2.z = 0;

	vertex v3_3;
		v3_3.x = 3;
		v3_3.y = 3*aur;
		v3_3.z = 0;

	vertex v3_4;
		v3_4.x = 3;
		v3_4.y = -3*aur;
		v3_4.z = 0;

	// Icosahedron generation
	ico.push_back(Triangle(v1_1, v3_4, v2_2));
	ico.push_back(Triangle(v2_2, v3_4, v2_3));
	ico.push_back(Triangle(v2_2, v2_3, v3_3));
	ico.push_back(Triangle(v3_3, v1_3, v3_2));
	ico.push_back(Triangle(v3_2, v2_4, v2_1));
	ico.push_back(Triangle(v2_1, v2_4, v3_1));
	ico.push_back(Triangle(v3_1, v1_1, v2_1));
	ico.push_back(Triangle(v1_1, v3_1, v3_4));
	ico.push_back(Triangle(v3_3, v1_3, v2_3));
	ico.push_back(Triangle(v3_2, v2_4, v1_3));

	ico.push_back(Triangle(v1_1, v2_2, v1_2));
	ico.push_back(Triangle(v1_2, v2_2, v3_3));
	ico.push_back(Triangle(v1_2, v3_3, v3_2));
	ico.push_back(Triangle(v1_2, v3_2, v2_1));
	ico.push_back(Triangle(v2_1, v1_1, v1_2));

	ico.push_back(Triangle(v1_4, v2_3, v3_4));
	ico.push_back(Triangle(v1_4, v3_4, v3_1));
	ico.push_back(Triangle(v2_4, v1_4, v3_1));
	ico.push_back(Triangle(v1_4, v2_4, v1_3));
	ico.push_back(Triangle(v1_3, v2_3, v1_4));
	return ico;
}

void drawFun(vector<Triangle> fig){
	glPushMatrix();
	/* Rotation control */
	glRotatef(last_angle_x + angle_x, 1, 0, 0);
	glRotatef(last_angle_z + angle_z, 0, 0, 1);
	/* !Rotation control */

	glColor3f(0.19, 0.31, 0.64);
	for(int i = 0; i < fig.size(); i++){
		glBegin(GL_LINES);
			glVertex3f(fig.at(i).v_list[0].x, fig.at(i).v_list[0].y, fig.at(i).v_list[0].z);
			glVertex3f(fig.at(i).v_list[1].x, fig.at(i).v_list[1].y, fig.at(i).v_list[1].z);

			glVertex3f(fig.at(i).v_list[1].x, fig.at(i).v_list[1].y, fig.at(i).v_list[1].z);
			glVertex3f(fig.at(i).v_list[2].x, fig.at(i).v_list[2].y, fig.at(i).v_list[2].z);

			glVertex3f(fig.at(i).v_list[2].x, fig.at(i).v_list[2].y, fig.at(i).v_list[2].z);
			glVertex3f(fig.at(i).v_list[0].x, fig.at(i).v_list[0].y, fig.at(i).v_list[0].z);
		glEnd();
	}
	glPopMatrix();
}

vector<Triangle> divideTriangle(Triangle t){
	// Getting the vertices that are already there
	vertex v1 = t.v_list.at(0);
	vertex v2 = t.v_list.at(1);
	vertex v3 = t.v_list.at(2);

	// Generating new vertices
	vertex v12;
		v12.x = (v1.x + v2.x)/2;
		v12.y = (v1.y + v2.y)/2;
		v12.z = (v1.z + v2.z)/2;

	vertex v23;
		v23.x = (v2.x + v3.x)/2;
		v23.y = (v2.y + v3.y)/2;
		v23.z = (v2.z + v3.z)/2;

	vertex v13;
		v13.x = (v1.x + v3.x)/2;
		v13.y = (v1.y + v3.y)/2;
		v13.z = (v1.z + v3.z)/2;

	vector<Triangle> sub_triangles;
	// Pushing the values to the return vector
	sub_triangles.push_back(Triangle(v1, v13, v12));
	sub_triangles.push_back(Triangle(v12, v13, v23));
	sub_triangles.push_back(Triangle(v23, v13, v3));
	sub_triangles.push_back(Triangle(v2, v12, v23));

	return sub_triangles;
}

void mesh(vector<Triangle> & fig){
	vector<Triangle> result;
	int i = 0;
	while(i < fig.size()){
		vector<Triangle> division = divideTriangle(fig.at(i));
		result.insert(result.end(), division.begin(), division.end());
		i++;
	}
	fig = result;
}

void project(vector<Triangle> & fig){
	for(int i = 0; i < fig.size(); i++){ // for each triangule in the figure
		for(int j = 0; j < 3; j++){ // for each vertex of a triangle
			// get the values separately
			float _x = fig.at(i).v_list.at(j).x;
			float _y = fig.at(i).v_list.at(j).y;
			float _z = fig.at(i).v_list.at(j).z;
			// Calculating the lambda
			// R = 1.9*3 (We multiplied every value by three at the beginning)
			float lambda = 1.9*3/( sqrt( pow(_x,2) + pow(_y,2) + pow(_z,2)) );
			// Aplying the "pushing" to the sphere we want
			fig.at(i).v_list.at(j).x *= lambda;
			fig.at(i).v_list.at(j).y *= lambda;
			fig.at(i).v_list.at(j).z *= lambda;
		}
	}
}
