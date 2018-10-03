#include "rasteriser.hpp"
#include "utilities/lodepng.h"
#include <vector>
#include <mpi.h>
#include <iomanip>
#include <chrono>
#include <limits>

struct fnCall {
    float3 fn_offset;
    float fn_scale;
    int fn_depth;
};



const std::vector<globalLight> lightSources = { {{0.3f, 0.5f, 1.0f}, {1.0f, 1.0f, 1.0f}} };

typedef struct perfCounter {
	unsigned long meshs = 0;
	unsigned long triagnles = 0;
} perfCounter;

perfCounter counter = {};

void runVertexShader( Mesh &mesh,
                      Mesh &transformedMesh,
                      float3 positionOffset,
                      float scale,
					  unsigned int const width,
					  unsigned int const height,
				  	  float const rotationAngle = 0)
{
	float const pi = std::acos(-1);
	// The matrices defined below are the ones used to transform the vertices and normals.

	// This projection matrix assumes a 16:9 aspect ratio, and an field of view (FOV) of 90 degrees.
	mat4x4 const projectionMatrix(
		0.347270,   0, 			0, 		0,
		0,	  		0.617370, 	0,		0,
		0,	  		0,			-1, 	-0.2f,
		0,	  		0,			-1,		0);

	mat4x4 translationMatrix(
		1,			0,			0,			0 + positionOffset.x /*X*/,
		0,			1,			0,			0 + positionOffset.y /*Y*/,
		0,			0,			1,			-10 + positionOffset.z /*Z*/,
		0,			0,			0,			1);

	mat4x4 scaleMatrix(
		scale/*X*/,	0,			0,				0,
		0, 			scale/*Y*/, 0,				0,
		0, 			0,			scale/*Z*/, 	0,
		0, 			0,			0,				1);

	mat4x4 const rotationMatrixX(
		1,			0,				0, 				0,
		0, 			std::cos(0), 	-std::sin(0),	0,
		0, 			std::sin(0),	std::cos(0), 	0,
		0, 			0,				0,				1);

	float const rotationAngleRad = (pi / 4.0f) + (rotationAngle / (180.0f/pi));

	mat4x4 const rotationMatrixY(
		std::cos(rotationAngleRad),		0,			std::sin(rotationAngleRad), 	0,
		0, 								1, 			0,								0,
		-std::sin(rotationAngleRad), 	0,			std::cos(rotationAngleRad), 	0,
		0, 								0,			0,								1);

	mat4x4 const rotationMatrixZ(
		std::cos(pi),	-std::sin(pi),	0,			0,
		std::sin(pi), 	std::cos(pi), 	0,			0,
		0,				0,				1,			0,
		0, 				0,				0,			1);

	mat4x4 const MVP =
		projectionMatrix * translationMatrix * rotationMatrixX * rotationMatrixY * rotationMatrixZ * scaleMatrix;

	for (unsigned int i = 0; i < mesh.vertices.size(); i++) {
		float4 currentVertex = mesh.vertices.at(i);
		float4 transformed = (MVP * currentVertex);
		currentVertex = transformed / transformed.w;
		currentVertex.x = (currentVertex.x + 0.5f) * (float) width;
		currentVertex.y = (currentVertex.y + 0.5f) * (float) height;
		transformedMesh.vertices.at(i) = currentVertex;
	}
}


void runFragmentShader( std::vector<unsigned char> &frameBuffer,
						unsigned int const baseIndex,
						Face const &face,
						float3 const &weights )
{
	float3 normal = face.getNormal(weights);

	float3 colour(0);
	for (globalLight const &l : lightSources) {
		float3 lightNormal = normal * l.direction;
		colour += (face.parent.material.Kd * l.colour) * (lightNormal.x + lightNormal.y + lightNormal.z);
	}

	colour = colour.clamp(0.0f, 1.0f);
	frameBuffer.at(4 * baseIndex + 0) = colour.x * 255.0f;
	frameBuffer.at(4 * baseIndex + 1) = colour.y * 255.0f;
	frameBuffer.at(4 * baseIndex + 2) = colour.z * 255.0f;
	frameBuffer.at(4 * baseIndex + 3) = 255;
}

/**
 * The main procedure which rasterises all triangles on the framebuffer
 * @param transformedMesh         Transformed mesh object
 * @param frameBuffer             frame buffer for the rendered image
 * @param depthBuffer             depth buffer for every pixel on the image
 * @param width                   width of the image
 * @param height                  height of the image
 */
void rasteriseTriangles( Mesh &transformedMesh,
                         std::vector<unsigned char> &frameBuffer,
                         std::vector<float> &depthBuffer,
                         unsigned int const width,
                         unsigned int const height )
{
	for (unsigned int i = 0; i < transformedMesh.faceCount(); i++) {

		Face face = transformedMesh.getFace(i);
		unsigned int minx = int(std::floor(std::min(std::min(face.v0.x, face.v1.x), face.v2.x)));
		unsigned int maxx = int(std::ceil (std::max(std::max(face.v0.x, face.v1.x), face.v2.x)));
		unsigned int miny = int(std::floor(std::min(std::min(face.v0.y, face.v1.y), face.v2.y)));
		unsigned int maxy = int(std::ceil (std::max(std::max(face.v0.y, face.v1.y), face.v2.y)));

		// Let's make sure the screen coordinates stay inside the window
		minx = std::max(minx, (unsigned int) 0);
		maxx = std::min(maxx, width);
		miny = std::max(miny, (unsigned int) 0);
		maxy = std::min(maxy, height);

		// We iterate over each pixel in the triangle's bounding box
		for(unsigned int x = minx; x < maxx; x++) {
			for(unsigned int y = miny; y < maxy; y++) {
				float u,v,w;
				if(face.inRange(x,y,u,v,w)){
					float pixelDepth = face.getDepth(u,v,w);
					if( pixelDepth >= -1 && pixelDepth <= 1 && pixelDepth < depthBuffer.at(y * width + x)) {
						depthBuffer.at(y * width + x) = pixelDepth;
						runFragmentShader(frameBuffer, x + (width * y), face, float3(u,v,w));
					}
				}
			}
		}
	}
}

void renderMeshFractal(
				std::vector<Mesh> &meshes,
				std::vector<Mesh> &transformedMeshes,
				unsigned int width,
				unsigned int height,
				std::vector<unsigned char> &frameBuffer,
				std::vector<float> &depthBuffer,
				float largestBoundingBoxSide,
				int depthLimit,
				float scale = 1.0,
				float3 distanceOffset = {0, 0, 0}) {

    // <STEP 2: get MPI_rank and comm_sz>
    int my_rank;
    int comm_sz;
    int flag;
    int work_rank; // the rank to which my_rank will give the current workload
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    work_rank = my_rank + 1;
    if (work_rank >= comm_sz){
        work_rank = 0;
    };
    // </STEP 2>

//
    int blockLen[3] {3, 1, 1};
    MPI_Datatype fn_send_recv;
    MPI_Datatype type[5] {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
    MPI_Aint disp[3];
    disp[0] = offsetof(fnCall, fn_offset);
    disp[1] = offsetof(fnCall, fn_scale);
    disp[2] = offsetof(fnCall, fn_depth);

    MPI_Type_create_struct(3, blockLen, disp, type, &fn_send_recv);
    MPI_Type_commit(&fn_send_recv);

//


    // Start by rendering the mesh at this depth
	for (unsigned int i = 0; i < meshes.size(); i++) {
		Mesh &mesh = meshes.at(i);
		Mesh &transformedMesh = transformedMeshes.at(i);
		runVertexShader(mesh, transformedMesh, distanceOffset, scale, width, height);
		rasteriseTriangles(transformedMesh, frameBuffer, depthBuffer, width, height);
	}

    // MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);

	// Check whether we've reached the recursive depth of the fractal we want to reach
	depthLimit--;
	//if(depthLimit == 0 && flag!=1) {
	//	return;
	//}


    if (depthLimit>=0){
        // Now we recursively draw the meshes in a smaller size
        for(int offsetX = -1; offsetX <= 1; offsetX++) {
            for(int offsetY = -1; offsetY <= 1; offsetY++) {
                for(int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                    float3 offset(offsetX,offsetY,offsetZ);
                    // We draw the new objects in a grid around the "main" one.
                    // We thus skip the location of the object itself.
                    if(offset == 0) {
                        continue;
                    }

                    float smallerScale = scale / 3.0;
                    float3 displacedOffset(
                            distanceOffset + offset * (largestBoundingBoxSide / 2.0f) * scale
                    );

                    // <STEP 3: distribute the work>
                    // std::cout << "sending work task from rank "<<my_rank<<" to  work rank "<<work_rank<<"  smallerScale = "<<smallerScale<<" depth= "<<depthLimit<<" offsetx="<<displacedOffset.x<< " offsety=" <<displacedOffset.y<<" offsetz="<<displacedOffset.z<<std::endl;

                    struct fnCall fn_call;
                    fn_call.fn_offset = displacedOffset;

                    fn_call.fn_scale = smallerScale;
                    fn_call.fn_depth = depthLimit;


                    MPI_Send(&fn_call, 1, fn_send_recv, work_rank, 0, MPI_COMM_WORLD);

                    work_rank = work_rank + 1;
                    if (work_rank >= comm_sz){
                        work_rank = 0;
                    };

                }
            }
        }

    }

    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    while (1==flag && 0 <= depthLimit){ //||
        // std::cout<<"Hi Im rank "<<my_rank<<" and Im looking for a job"<<std::endl;
        // search for new work
        struct fnCall fn_call;
        MPI_Recv(&fn_call, 1, fn_send_recv, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        float sscale = fn_call.fn_scale;
        int dlimit = fn_call.fn_depth;
        float3 doffset = fn_call.fn_offset;
        // std::cout << "Hi I'm rank "<<my_rank<<" and received work,  smallerScale = "<<sscale<<" depth= "<<dlimit<<" offsetx="<<doffset.x<< " offsety=" <<doffset.y<<" offsetz="<<doffset.z<<std::endl;
        renderMeshFractal(meshes, transformedMeshes, width, height, frameBuffer, depthBuffer, largestBoundingBoxSide, dlimit, sscale, doffset);
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }
    if(depthLimit == 0) {
        return;
    }

    // </STEP 3>

}

// This function kicks off the rasterisation process.
std::vector<unsigned char> rasterise(std::vector<Mesh> &meshes, unsigned int width, unsigned int height, unsigned int depthLimit) {

    // <STEP 1: initialising the MPI>
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // </STEP 1>

    int blockLen[3] {3, 1, 1};
    MPI_Datatype fn_send_recv;
    MPI_Datatype type[5] {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
    MPI_Aint disp[3];
    disp[0] = offsetof(fnCall, fn_offset);
    disp[1] = offsetof(fnCall, fn_scale);
    disp[2] = offsetof(fnCall, fn_depth);

    MPI_Type_create_struct(3, blockLen, disp, type, &fn_send_recv);
    MPI_Type_commit(&fn_send_recv);


	// We first need to allocate some buffers.
	// The framebuffer contains the image being rendered.
	std::vector<unsigned char> frameBuffer;
	// The depth buffer is used to make sure that objects closer to the camera occlude/obscure objects that are behind it
	std::vector<float> depthBuffer;
	frameBuffer.resize(width * height * 4, 0);
	for (unsigned int i = 3; i < (4 * width * height); i+=4) {
		frameBuffer.at(i) = 255;
	}
	depthBuffer.resize(width * height, 1);

	float3 boundingBoxMin(std::numeric_limits<float>::max());
	float3 boundingBoxMax(std::numeric_limits<float>::min());

	std::cout << "Rendering image... " << std::flush;

	std::vector<Mesh> transformedMeshes;
	for(unsigned int i = 0; i < meshes.size(); i++) {
		transformedMeshes.push_back(meshes.at(i).clone());

		for(unsigned int vertex = 0; vertex < meshes.at(i).vertices.size(); vertex++) {
			boundingBoxMin.x = std::min(boundingBoxMin.x, meshes.at(i).vertices.at(vertex).x);
			boundingBoxMin.y = std::min(boundingBoxMin.y, meshes.at(i).vertices.at(vertex).y);
			boundingBoxMin.z = std::min(boundingBoxMin.z, meshes.at(i).vertices.at(vertex).z);

			boundingBoxMax.x = std::max(boundingBoxMax.x, meshes.at(i).vertices.at(vertex).x);
			boundingBoxMax.y = std::max(boundingBoxMax.y, meshes.at(i).vertices.at(vertex).y);
			boundingBoxMax.z = std::max(boundingBoxMax.z, meshes.at(i).vertices.at(vertex).z);
		}
	}

	float3 boundingBoxDimensions = boundingBoxMax - boundingBoxMin;
	float largestBoundingBoxSide = std::max(std::max(boundingBoxDimensions.x, boundingBoxDimensions.y), boundingBoxDimensions.z);

    if (my_rank ==0){
        renderMeshFractal(meshes, transformedMeshes, width, height, frameBuffer, depthBuffer, largestBoundingBoxSide, depthLimit);//depthLimit);
    }
    else{
        struct fnCall fn_call;
        MPI_Recv(&fn_call, 1, fn_send_recv ,MPI_ANY_SOURCE , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        int depthLimit = fn_call.fn_depth;
        float smallerScale = fn_call.fn_scale;
        float3 distanceOffset = fn_call.fn_offset;
        // std::cout <<"I received this smallerScale "<< smallerScale<<" Hade bra!" << std::endl;

        /// std::cout << "I am rank " <<my_rank<<" and Im the game right now!" << std::endl;
        // std::cout <<"I received this depthLimit "<< depthLimit<<" Hade bra!" << std::endl;
        // std::cout <<"I received this offsets "<< distanceOffset.x<<" "<< distanceOffset.y<<" "<<distanceOffset.z <<  " Hade bra!" << std::endl;



        renderMeshFractal(meshes, transformedMeshes, width, height, frameBuffer, depthBuffer, largestBoundingBoxSide, depthLimit, smallerScale, distanceOffset);//depthLimit);

    }

	float* depthbuff = new float[width*height];
	float* fin_depthbuff = new float[width*height];
	int i;
	for(i = 0; i < width*height; ++i) {
		depthbuff[i] = depthBuffer.at(i);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Allreduce(depthbuff, fin_depthbuff, width*height, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);


	std::vector<unsigned char> fin_frameBuffer;
	fin_frameBuffer.resize(width * height * 4, 0);
	for (unsigned int i = 3; i < (4 * width * height); i+=4) {
		fin_frameBuffer.at(i) = 0;
	}
	for(i = 0; i < height*width; ++i) {
		if(fin_depthbuff[i] == depthbuff[i]) {
			fin_frameBuffer.at(i) = frameBuffer.at(i);
		}
	}
	unsigned char* temp_fin_frameBuffer = new unsigned char[frameBuffer.size()];
	unsigned char* temp_frameBuffer = new unsigned char[frameBuffer.size()];


	for(i = 0; i < frameBuffer.size(); ++i) {
		temp_frameBuffer[i] = fin_frameBuffer.at(i);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Reduce(temp_frameBuffer, temp_fin_frameBuffer, width*height, MPI_CHAR, MPI_BOR, 0, MPI_COMM_WORLD);

	for(i = 0; i < frameBuffer.size(); ++i) {
		fin_frameBuffer.at(i) = temp_fin_frameBuffer[i];
	}

	delete[] depthbuff;
	delete[] fin_depthbuff;
	delete[] temp_fin_frameBuffer;
	delete[] temp_frameBuffer;

	std::cout << "finished!" << std::endl;

	return fin_frameBuffer;


}
