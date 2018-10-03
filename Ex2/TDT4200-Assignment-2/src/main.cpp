#include <iostream>
#include <cstring>
#include <string>
#include "utilities/OBJLoader.hpp"
#include "utilities/lodepng.h"
#include "rasteriser.hpp"
#include <mpi.h>

int main(int argc, char **argv) {
	std::string input("../input/spheres.obj");
	std::string output("../output/spheres.png");
	unsigned int width = 1920;
	unsigned int height = 1080;
	unsigned int depth = 3;

    int comm_sz; // size of the communicator
    int my_rank; // rank of the individual process

    MPI_Init(NULL, NULL); // ronald: setting up MPI environment

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); // getting the size of the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // getting the individual rank

    std::cout << "Hi there! I'm process number " << my_rank << " of the "<<comm_sz<<"!"<<std::endl;

	for (int i = 1; i < argc; i++) {
		if (i < argc -1) {
			if (std::strcmp("-i", argv[i]) == 0) {
				input = argv[i+1];
			} else if (std::strcmp("-o", argv[i]) == 0) {
				output = argv[i+1];
			} else if (std::strcmp("-w", argv[i]) == 0) {
				width = (unsigned int) std::stoul(argv[i+1]);
			} else if (std::strcmp("-h", argv[i]) == 0) {
				height = (unsigned int) std::stoul(argv[i+1]);
			} else if (std::strcmp("-d", argv[i]) == 0) {
				depth = (int) std::stoul(argv[i+1]);
			}
		}
	}
	std::vector<Mesh> meshs;
    output = "output/file_nb_" + std::to_string(my_rank) + ".png";
	int scale = 1;

	// Creating float4 and float 3 MPI data structs
	int blockLen_f4[4] = {1, 1, 1, 1};
	int blockLen_f3[3] = {1, 1, 1};
	MPI_Datatype mpi_float4;
	MPI_Datatype mpi_float3;
	MPI_Datatype type4[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
	MPI_Datatype type3[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
	MPI_Aint disp4[4];
	MPI_Aint disp3[3];

	disp4[0] = offsetof(float4, x); disp4[1] = offsetof(float4, y); disp4[2] = offsetof(float4, z); disp4[3] = offsetof(float4, w);
	disp3[0] = offsetof(float3, x); disp3[1] = offsetof(float3, y); disp3[2] = offsetof(float3, z);

	MPI_Type_create_struct(4, blockLen_f4, disp4, type4, &mpi_float4);
	MPI_Type_create_struct(3, blockLen_f3, disp3, type3, &mpi_float3);
	MPI_Type_commit(&mpi_float4);
	MPI_Type_commit(&mpi_float3);

	if(my_rank == 0) {
		// Loading the mesh object
		meshs = loadWavefront(input, false);
		// Sending the broadcast
		for(unsigned int i = 0; i < meshs.size(); ++i) {
			MPI_Bcast(&meshs[i].vertices[i], meshs[i].vertices.size(), mpi_float4, 0, MPI_COMM_WORLD);
			MPI_Bcast(&meshs[i].textures[i], meshs[i].textures.size(), mpi_float3, 0, MPI_COMM_WORLD);
			MPI_Bcast(&meshs[i].normals[i], meshs[i].normals.size(), mpi_float3, 0, MPI_COMM_WORLD);
		}
	}
	else {
		// Loading the mesh object
		meshs = loadWavefront(input, false);
		// Removing the contents from vertices, textures and normals
		for(unsigned int i = 0; i < meshs.size(); ++i) {
			for(unsigned int v = 0; v < meshs.at(i).vertices.size(); ++v) {
				meshs.at(i).vertices.at(v).x = 0;
				meshs.at(i).vertices.at(v).y = 0;
				meshs.at(i).vertices.at(v).z = 0;
				meshs.at(i).vertices.at(v).w = 0;
			}
			for(unsigned int t = 0; t < meshs.at(i).textures.size(); ++t) {
				meshs.at(i).textures.at(t).x = 0;
				meshs.at(i).textures.at(t).y = 0;
				meshs.at(i).textures.at(t).z = 0;
			}
			for(unsigned int n = 0; n < meshs.at(i).normals.size(); ++n) {
				meshs.at(i).normals.at(n).x = 0;
				meshs.at(i).normals.at(n).y = 0;
				meshs.at(i).normals.at(n).z = 0;
			}
		}
		// Receiving the broadcast
		for(unsigned int i = 0; i < meshs.size(); ++i) {
			MPI_Bcast(&meshs[i].vertices[i], meshs[i].vertices.size(), mpi_float4, 0, MPI_COMM_WORLD);
			MPI_Bcast(&meshs[i].textures[i], meshs[i].textures.size(), mpi_float3, 0, MPI_COMM_WORLD);
			MPI_Bcast(&meshs[i].normals[i], meshs[i].normals.size(), mpi_float3, 0, MPI_COMM_WORLD);

		}
	}

	std::cout << "Loading '" << input << "' file... " << std::endl;
	std::vector<unsigned char> frameBuffer = rasterise(meshs, scale*width, scale*height, depth);

	std::cout << "Writing image to '" << output << "'..." << std::endl;
	unsigned error = lodepng::encode(output, frameBuffer, width, height);

	if(error)
	{
		std::cout << "An error occurred while writing the image file: " << error << ": " << lodepng_error_text(error) << std::endl;
	}
    MPI_Finalize();
	return 0;
}
