#include <iostream>
#include <cstring>
#include <string>
#include "utilities/OBJLoader.hpp"
#include "utilities/lodepng.h"
#include "rasteriser.hpp"
#include <mpi.h>

int main(int argc, char **argv) {
	std::string input("../input/sphere.obj");
	std::string output("../output/sphere.png");
	unsigned int width = 1920;
	unsigned int height = 1080;
	unsigned int depth = 3;
    int scale;

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
	std::string my_rank_string = std::to_string(my_rank);
    output = "output/file_nb_" + my_rank_string +".png";
	if(my_rank == 0) {
		for(int i = 1; i < comm_sz; ++i) {
			scale = i+1; // just declare something different for all processess
			MPI_Send(
					&scale, // pointer to stored data
					1, // packet size
					MPI_INT, // packet type
					i, // receiver
					0, // tag
					MPI_Comm MPI_COMM_WORLD);
            // communicator
		}
	}
	else {
		MPI_Recv(
				&scale,
				1,
				MPI_INT,
				0,
				0,
				MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
	}



	std::cout << "Loading '" << input << "' file... " << std::endl;

	std::vector<Mesh> meshs = loadWavefront(input, false);

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
