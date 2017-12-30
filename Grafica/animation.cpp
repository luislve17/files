#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <GL/glut.h>
#include <iostream>
#include <cmath>

using namespace std;

class Column{
public:
	int x, y, z, w, h, t;
	float clock_val;
	Column(int x, int y, int z, int w, int h, int t, int cv);
	void draw();
};

Column::Column(int x, int y, int z, int w, int h, int t, int cv){
	this->x = x;this->y = y;this->z = z;this->w = w;this->h = h;this->t = t;this->clock_val=cv;
}

void Column::draw(){
	glPushMatrix();
	glRotatef(60, 1,0,0);
	glRotatef(135, 0,0,1);
	glBegin(GL_QUADS);
		glColor3f(1.0, 1.0, 1.0);
		glVertex3f(x-(float)w/2.0, y-(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x+(float)w/2.0, y-(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x+(float)w/2.0, y+(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x-(float)w/2.0, y+(float)h/2.0, z-(float)t/2.0);

		glColor3f(0.38, 0.55, 0.91);
		glVertex3f(x-(float)w/2.0, y-(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x-(float)w/2.0, y+(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x-(float)w/2.0, y+(float)h/2.0, z+(float)t/2.0);
		glVertex3f(x-(float)w/2.0, y-(float)h/2.0, z+(float)t/2.0);

		glColor3f(0.24, 0.83, 0.64);
		glVertex3f(x+(float)w/2.0, y+(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x-(float)w/2.0, y+(float)h/2.0, z-(float)t/2.0);
		glVertex3f(x-(float)w/2.0, y+(float)h/2.0, z+(float)t/2.0);
		glVertex3f(x+(float)w/2.0, y+(float)h/2.0, z+(float)t/2.0);
	glEnd();
	glPopMatrix();
}

vector<Column> cols_vector;

int main(int argc, char ** argv){
	sf::RenderWindow w(sf::VideoMode(640, 360), "Animation");
	w.setFramerateLimit(60);

	/* OpenGl settings */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-100.0, 100.0, -100.0, 100.0, -200.0, 200.0);
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
	/* !OpenGl settings */

	sf::Clock clock;

	for(int i = -10; i <= 10; i++){
		for(int j = -4; j <= 4; j++){
			cols_vector.push_back(Column(-6*i,8*j,0,5,5,60,i));
		}
	}

	while(w.isOpen()){
		sf::Event e;
		while(w.pollEvent(e)){
			if(e.type == sf::Event::Closed) {
				w.close();
			}
			else if (e.type == sf::Event::Resized)
            {
				w.setView(sf::View(sf::FloatRect(0.f, 0.f,
					                static_cast<float>(w.getSize().x),
					                static_cast<float>(w.getSize().y))
								)
							);
            }
		}

		/* OpenGL*/
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		glClearColor(1.0,1.0,1.0,1.0);

		for(int i = 0; i < cols_vector.size(); i++){
			cols_vector.at(i).draw();
		}

		for(int i = 0; i < cols_vector.size(); i++){
			cols_vector.at(i).t = 60 + 25*sin(cols_vector.at(i).clock_val);
			cols_vector.at(i).clock_val += .1;
		}

		/* !OpenGL */
		w.display();
	}
}
