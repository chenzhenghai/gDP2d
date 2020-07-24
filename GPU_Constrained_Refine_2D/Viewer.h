#pragma once
#include "triangle.h"
#include "freeglut\freeglut.h"

triangulateio *draw_result;

void drawTriangles()
{
	if (draw_result->trianglelist != NULL)
	{

		for (int i = 0; i<draw_result->numberoftriangles; i++)
		{


			int index_1, index_2, index_3;
			double x_1, y_1, x_2, y_2, x_3, y_3;

			index_1 = draw_result->trianglelist[3 * i];
			index_2 = draw_result->trianglelist[3 * i + 1];
			index_3 = draw_result->trianglelist[3 * i + 2];

			x_1 = draw_result->pointlist[2 * index_1];
			y_1 = draw_result->pointlist[2 * index_1 + 1];
			x_2 = draw_result->pointlist[2 * index_2];
			y_2 = draw_result->pointlist[2 * index_2 + 1];
			x_3 = draw_result->pointlist[2 * index_3];
			y_3 = draw_result->pointlist[2 * index_3 + 1];

			glColor3f(0.0, 0.0, 0.0);
			glLineWidth(1);
			glBegin(GL_LINES);
			glVertex2f(x_1, y_1);
			glVertex2f(x_2, y_2);
			glVertex2f(x_2, y_2);
			glVertex2f(x_3, y_3);
			glVertex2f(x_3, y_3);
			glVertex2f(x_1, y_1);
			glEnd();
		}
	}
}

void drawSegments()
{
	if (draw_result->segmentlist != NULL)
	{
		glColor3f(0.0, 0.0, 0.0);
		for (int i = 0; i<draw_result->numberofsegments; i++)
		{
			int index_1, index_2;
			double x_1, y_1, x_2, y_2;

			index_1 = draw_result->segmentlist[2 * i];
			index_2 = draw_result->segmentlist[2 * i + 1];

			x_1 = draw_result->pointlist[2 * index_1];
			y_1 = draw_result->pointlist[2 * index_1 + 1];
			x_2 = draw_result->pointlist[2 * index_2];
			y_2 = draw_result->pointlist[2 * index_2 + 1];

			if (i == -1)
			{
				glColor3f(0.0, 0.0, 1.0);
				glLineWidth(2);
				glBegin(GL_LINES);
				glVertex2f(x_1, y_1);
				glVertex2f(x_2, y_2);
				glEnd();
			}
			else
			{
				glColor3f(1.0, 0.0, 0.0);
				glLineWidth(1);
				glBegin(GL_LINES);
				glVertex2f(x_1, y_1);
				glVertex2f(x_2, y_2);
				glEnd();
			}
		}
	}
}

void drawPoints()
{
	if (draw_result->pointlist != NULL)
	{
		glPointSize(3);
		for (int i = 0; i<draw_result->numberofpoints; i++)
		{
			double x = draw_result->pointlist[2 * i];
			double y = draw_result->pointlist[2 * i + 1];

			glColor3f(1.0, 0.0, 0.0);

			glBegin(GL_POINTS);
			glVertex2f(x, y);
			glEnd();
		}
	}
}

void drawSegmentsPoints()
{
	if (draw_result->segmentlist != NULL)
	{
		glPointSize(0.1);
		for (int i = 0; i<draw_result->numberofsegments; i++)
		{
			int index_1, index_2;
			double x_1, y_1, x_2, y_2;

			index_1 = draw_result->segmentlist[2 * i];
			index_2 = draw_result->segmentlist[2 * i + 1];

			x_1 = draw_result->pointlist[2 * index_1];
			y_1 = draw_result->pointlist[2 * index_1 + 1];
			x_2 = draw_result->pointlist[2 * index_2];
			y_2 = draw_result->pointlist[2 * index_2 + 1];

			glColor3f(1.0, 0.0, 0.0);
			glBegin(GL_POINTS);
			glVertex2f(x_1, y_1);
			glVertex2f(x_2, y_2);
			glEnd();
		}
	}
}

void init(void)
{
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glShadeModel(GL_FLAT);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glTranslatef(50.0f,50.0f,0.0f);
	glScalef(2.8f,2.8f,0.f);
	drawTriangles();
	drawSegments();
	//drawPoints();
	//drawSegmentsPoints();

	glFlush();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
}

void drawTriangulation(int argc, char** argv, triangulateio * input)
{
	draw_result = input;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMainLoop();
}