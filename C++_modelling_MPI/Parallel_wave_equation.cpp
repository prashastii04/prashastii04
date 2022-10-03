#include <mpi.h>
#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdlib>


#define DO_TIMING

using namespace std;

//------------------------------------------------------
// PARALLELISING THE WAVE EQUATION
// In this code, the library MPI is used to parallelise the wave equation 
// using periodic/no periodic boundary conditions (by default periodic conditions 
// are assigned)

int p, id;

vector<int> neighbour_ids;

int tag_num=1;
const bool periodic = true;

// int num_points = 100;
int *process_chunk, *row_start, *col_start, *row_end, *col_end, *imax_proc_list, *jmax_proc_list;

//  imax = rows in the larger grid
//  jmax = cols in the larger grid
int imax = 500, jmax = 500;
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.04, dt;
double y_max = 10.0, x_max = 10.0;
double c = 1;
int left_neighbour_id, right_neighbour_id, top_neighbour_id, bottom_neighbour_id;

// number of rows and cols
int no_of_rows,  no_of_columns;

// id row and id columns
int id_row, id_column;

// create 4 send 4 receive datatypes for the different communications
MPI_Datatype Datatype_left_send, Datatype_right_send, Datatype_top_send, Datatype_bottom_send, Datatype_left_recv, Datatype_right_recv, Datatype_top_recv, Datatype_bottom_recv;

// allocate grids
double *grid_1D = nullptr;
double **grid_2D= nullptr;
double  *old_grid_1D = nullptr;
double **old_grid_2D = nullptr;
double *new_grid_1D = nullptr;
double **new_grid_2D = nullptr;

// the dx and dy
double dx = x_max / ((double)jmax - 1);
double dy = y_max / ((double)imax - 1);

// imax_proc = rows per process 
// jmax_proc = cols per process 
int imax_proc, jmax_proc;

// creating variables for the timings
double start_time, end_time;
// initialise iteration ad output count (for storing the grid)
int out_cnt = 0, it = 0;
//------------------------------------------------------
// function to divide the processes into rows and cols
void find_domain(int p, int &rows, int &columns)		//A bit brute force - this can definitely be made more efficient!
{
	int min_gap = p;
	int top = sqrt(p) + 1;
	for (int i = 1; i <= top; i++)
	{
		if (p % i == 0)
		{
			int gap = abs(p / i - i);

			if (gap < min_gap)
			{
				min_gap = gap;
				no_of_rows = i;
				no_of_columns = p / i;
			}
		}
	}

	// if (id == 0)
	// 	cout << "Divide " << p << " into " << rows << " by " << no_of_columns << " grid" << endl;
}
// ------------------------------------------------------------------
// the following function gets the column and row number from the id
void id_to_index(int id, int &id_row, int &id_column)
{
	id_column = id % no_of_columns;
	id_row = id / no_of_columns;
}
//------------------------------------------------------------------
// the following function gets the id from the column and row number
int id_from_index(int id_row, int id_column)
{
	if (id_row >= no_of_rows || id_row<0)
		return -1;
	if (id_column >= no_of_columns || id_column<0)
		return -1;

	return id_row*no_of_columns + id_column;
}
//---------------------------------------------------------------------
// create 8 different datatypes - 4 send and 4 receive
void createdatatypes(double** data, int m, int n)
{
	// creating send data types
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	MPI_Aint add_start;
	add_start=0;
	MPI_Aint temp_address;

	// createdatatypes(grid_2D,imax_proc,jmax_proc);

	//left send
	for (int i = 0; i < m-1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[i+1][n], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(&data[0][0], &add_start);
	for (int i = 0; i < m; i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m-2, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_left_send);
	MPI_Type_commit(&Datatype_left_send);

	//right send to process on left
	block_lengths.resize(0);
	typelist.resize(0);
	addresses.resize(0);
	for (int i = 1; i < m-1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[n-1][i+1], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(&data[0][0], &add_start);
	for (size_t i = 0; i < addresses.size(); i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m - 2, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_right_send);
	MPI_Type_commit(&Datatype_right_send);

	// top send
	int block_length = n-2;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	MPI_Get_address(&data[1][1], &address);
	MPI_Get_address(&data[0][0], &add_start);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_top_send);
	MPI_Type_commit(&Datatype_top_send);

	// bottom send
	MPI_Get_address(&data[1][n], &address);
	MPI_Get_address(&data[0][0], &add_start);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_bottom_send);
	MPI_Type_commit(&Datatype_bottom_send);

	// left receive - getting data from right processor
	block_lengths.resize(0);
	typelist.resize(0);
	addresses.resize(0);
	vector<MPI_Aint> offset;
	for (int i = 0; i < m-1 ; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&data[i+1][0], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(&data[0][0], &add_start);
	for (int i = 0; i < addresses.size(); i++)
		offset.push_back(addresses[i] - add_start);
	MPI_Type_create_struct(m - 2, block_lengths.data(), offset.data(), typelist.data(), &Datatype_left_recv);
	MPI_Type_commit(&Datatype_left_recv);


	//right receive - getting data from left processor
	block_lengths.resize(0);
	typelist.resize(0);
	addresses.resize(0);
	for (int i = 0; i < m-1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Get_address(&data[i+1][n-1], &temp_address);
		addresses.push_back(temp_address);
	}
	
	MPI_Get_address(&data[0][0], &add_start);
	for (int i = 0; i < addresses.size(); i++)
		offset.push_back(addresses[i] - add_start);

	MPI_Type_create_struct(m - 2, block_lengths.data(), offset.data(), typelist.data(), &Datatype_right_recv);
	MPI_Type_commit(&Datatype_right_recv);

	// //top recv
	MPI_Get_address(&data[0][1], &address);
	MPI_Get_address(&data[0][0], &add_start);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_top_recv);
	MPI_Type_commit(&Datatype_top_recv);

	//bottom recv
	MPI_Get_address(&data[n+1][1], &address);
	MPI_Get_address(&data[0][0], &add_start);
	address = address - add_start;
	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_bottom_recv);
	MPI_Type_commit(&Datatype_bottom_recv);

	
}
//---------------------------------------------------------------------
// this function finds the id of the different neighbours of the current id
void find_neighbours(int &left_neighbour_id, int &right_neighbour_id, int &top_neighbour_id, int &bottom_neighbour_id)
{
	find_domain(p,no_of_rows,no_of_columns);
    id_to_index(id,id_row,id_column);

 
   
	if(periodic)
	{
		right_neighbour_id=id+1;
		left_neighbour_id=id-1;
		top_neighbour_id=id+no_of_columns;
		bottom_neighbour_id=id-no_of_columns;
		if (id_row==0)
		{
			bottom_neighbour_id=id+(no_of_rows-1)*no_of_columns;
		}
		if(id_column==0)
		{
			left_neighbour_id=id+(no_of_columns-1);
			
		}
		if(id_row==no_of_rows-1)
		{
			top_neighbour_id=(id+(no_of_rows-1)*no_of_columns)%no_of_columns;
			
			
		}
		if(id_column==no_of_columns-1)
		{
			right_neighbour_id=id-(no_of_columns-1);
			
		}
	}	
	else{
		right_neighbour_id=id+1;
		left_neighbour_id=id-1;
		top_neighbour_id=id+no_of_columns;
		bottom_neighbour_id=id-no_of_columns;

		if (id_row==0)
		{
			bottom_neighbour_id=-1;
			
		}
		if(id_column==0)
		{
			left_neighbour_id=-1;
			
		}
		if(id_row==no_of_rows-1)
		{
			top_neighbour_id=-1;
			
		}
		if(id_column==no_of_columns-1)
		{
			right_neighbour_id=-1;
			
		}
		
	}

	neighbour_ids.push_back(left_neighbour_id);
	neighbour_ids.push_back(top_neighbour_id);
	neighbour_ids.push_back(right_neighbour_id);
	neighbour_ids.push_back(bottom_neighbour_id);

	// cout << " Neighbour list for process " << id << ": " << endl;
	// cout << "row id" << id_row << endl;
	// cout << "column id" << id_column << endl;
	// cout << "id "<< id << " left top right bottom" << endl;
	// for (int neigh_id : neighbour_ids)
	// 	cout << "\t" << neigh_id;
	// cout << endl;
	// cout.flush();

}
//---------------------------------------------------------------------
// this function calculates the next time step of the equation using a FTCS scheme
void do_iteration(void)
{
	//Calculate the new displacement for all the points not on the boundary of the domain
	//Note that in parallel the edge of processor's region is not necessarily the edge of the domain
	for (int i = 1; i < imax_proc +1; i++)
		for (int j = 1; j < jmax_proc+1  ; j++)
			new_grid_2D[i][j] = pow(dt * c, 2.0) * ((grid_2D[i + 1][j] - 2.0 * grid_2D[i][j] + grid_2D[i - 1][j]) / pow(dx, 2.0) + (grid_2D[i][j + 1] - 2.0  * grid_2D[i][j] + grid_2D[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid_2D[i][j] - old_grid_2D[i][j];

	// // Implement boundary conditions - This is a Neumann boundary that I have implemented
	// for (int i = 1; i < imax_proc+1; i++)
	// {
	// 	new_grid_2D[i][0] = new_grid_2D[i][1];
	// 	new_grid_2D[i][jmax - 1] = new_grid_2D[i][jmax - 2];
	// }

	// for (int j = 1; j < jmax_proc+1; j++)
	// {
	// 	new_grid_2D[0][j] = new_grid_2D[1][j];
	// 	new_grid_2D[imax-1][j] = new_grid_2D[imax-2][j];
	// }

	t += dt;

	//Note that I am not copying data between the grid_2Ds, which would be very slow, but rather just swapping pointers
	// swap(old_grid_2D,new_grid_2D);
	// swap(old_grid_2D,grid_2D);
	
	// for(int i=1; i<imax_proc+1; i++)
	// {
	// 	// for(int j=1; j<jmax_proc+1; j++)
	// 	// {
	// 	// double* temp_1D[i][j] = old_grid_1D[i][j];
	// 	// old_grid_1D[i][j] = grid_1D[i][j];
	// 	// grid_1D[i][j] = new_grid_1D[i][j];
	// 	// new_grid_1D[i][j] = temp_1D;
		
	// 	double* temp_2D = old_grid_2D[i];
	// 	old_grid_2D[i] = grid_2D[i];
	// 	grid_2D[i] = new_grid_2D[i];
	// 	new_grid_2D[i] = &temp_2D[i];
		
	// }

	for(int i=1; i<imax_proc+1; i++)
	{
		// for(int j=1; j<jmax_proc+1; j++)
		// {
		// double* temp_1D[i][j] = old_grid_1D[i][j];
		// old_grid_1D[i][j] = grid_1D[i][j];
		// grid_1D[i][j] = new_grid_1D[i][j];
		// new_grid_1D[i][j] = temp_1D;
		
		double* temp_grid_2D = old_grid_2D[i];
		old_grid_2D[i] = grid_2D[i];
		grid_2D[i] = new_grid_2D[i];
		new_grid_2D[i] = temp_grid_2D;
		
	}
	
}
//---------------------------------------------------------------------
// this function assigns an initial condition to the grid - Here neumann conditions are also applied
void initial_condition()
{

    double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;
	// for (int i = 1; i < imax_proc - 1; i++)
	// 	for (int j = 1; j < jmax_proc - 1; j++)
	// 	{
	// 	
	// 			double h = 5.0*(cos(dist / r_splash * M_PI) + 1.0);
	// 			grid_2D[i*jmax_proc][j] = h;
	// 			old_grid_2D[i*jmax_proc][j] = h;
	// 		}
	// 		cout <<"process: " << id  << " grid: " << grid_2D;
	// 	}
	
	for (int i=1;i<imax_proc+1;i++)
	{
		for (int j=1;j<jmax_proc+1;j++)
		{
			double x = dx * i*(1+id_column);
			double y = dy * j*(1+id_row);
			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));
			if (dist < r_splash)
	 		{
			
			double h = 5.0*(cos(dist / r_splash * M_PI) + 1.0);
				grid_2D[i][j]=h;
				old_grid_2D[i][j]=h;
				// new_grid_2D[i][j]=rand();
			}
		}
	}

		
		
}
//---------------------------------------------------------------------
// this function prints the grid to dat file output
void grid_to_file(int out)
{
	//Write the output for a single time step to file
	stringstream fname;
	fstream f1;
	fname << "./out_parallel/output" << " process_" << id << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 1; i < imax_proc+1; i++)
	{
		for (int j = 1; j < jmax_proc+1; j++)
			f1 << grid_2D[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
}

//---------------------------------------------------------------------
// this function creates the initial grid for the start
void create_grid()
{
	grid_1D = new double[(imax_proc+2)*(jmax_proc+2)];
	grid_2D = new double*[(imax_proc+2)];

	old_grid_1D = new double[(imax_proc+2)*(jmax_proc+2)];
	old_grid_2D = new double*[(imax_proc+2)];

	new_grid_1D = new double[(imax_proc+2)*(jmax_proc+2)];
	new_grid_2D = new double*[(imax_proc+2)];

	for (int i = 0; i < imax_proc+2; i++)
		grid_2D[i] = &grid_1D[i * (jmax_proc+2)];

	for (int i = 0; i < imax_proc+2; i++)
		old_grid_2D[i] = &old_grid_1D[i * (jmax_proc+2)];

	for (int i = 0; i < imax_proc+2; i++)
		new_grid_2D[i] = &new_grid_1D[i * (jmax_proc+2)];
}
//---------------------------------------------------------------------
// this function initialises all the grid values to zero
void initialise_grid(double** grid_2D, double** new_grid_2D, double** old_grid_2D)
{
	for (int i=0;i<imax_proc+2;i++)
	{
		for (int j=0;j<jmax_proc+2;j++)
		{
				grid_2D[i][j]=0;
				old_grid_2D[i][j]=0;
				new_grid_2D[i][j]=0;
		}
	}
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
// Here the timings are recorded
#ifdef DO_TIMING

	auto start = chrono::high_resolution_clock::now();

#endif
// this finds the domain based on processes, and finds its neighbours
    find_domain(p, no_of_rows, no_of_columns);
    id_to_index(id, id_row, id_column);
	find_neighbours(left_neighbour_id, right_neighbour_id, top_neighbour_id, bottom_neighbour_id);
	
// calculate the number of points assigned to each process locally
	imax_proc=imax/no_of_rows;
	jmax_proc = jmax/no_of_columns;

// create the grids - current, new and old grids
	create_grid();

// initialising the grid to 0
	initialise_grid(grid_2D,old_grid_2D,new_grid_2D);

// create 8 data types - 4 send and 4 receive
	createdatatypes(grid_2D,(imax_proc),(jmax_proc));

// initialise iteration number	
	it=0;
// set initial condition
	initial_condition();
// increment dt and iteration number
	t_out += dt_out;
	t = 0.0;
	dt = 0.1 * min(dx, dy) / c;
	it++;

// loop through for each time step
	while (t<t_max)
		{

			MPI_Request* request = nullptr;
			request = new MPI_Request[8];

			int count = 0;
			// sending the data to the 4 neighbouring processes
			MPI_Isend(&grid_2D[0][0],1,Datatype_left_send,left_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			MPI_Isend(&grid_2D[0][0],1,Datatype_right_send,right_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			MPI_Isend(&grid_2D[0][0],1,Datatype_top_send,top_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			MPI_Isend(&grid_2D[0][0],1,Datatype_bottom_send,bottom_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;

			// receiving the data from the 4 neighbouring processes
			MPI_Irecv(&grid_2D[0][0],1,Datatype_left_recv,left_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			MPI_Irecv(&grid_2D[0][0],1,Datatype_right_recv,right_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			MPI_Irecv(&grid_2D[0][0],1,Datatype_top_recv,top_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			MPI_Irecv(&grid_2D[0][0],1,Datatype_bottom_recv,bottom_neighbour_id,tag_num,MPI_COMM_WORLD,&request[count]);
			count++;
			

			// wait for all processes to finish
			MPI_Waitall(count, request, MPI_STATUSES_IGNORE);

			// increment tag
			tag_num++;
			
			// iterate 
			do_iteration();


			// cout a grid and increment t
			if (t_out <= t)
			{
				cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
				grid_to_file(out_cnt);
				out_cnt++;
				t_out += dt_out;
			}
		
			it++;
			delete[] request;
		
		}

	MPI_Barrier(MPI_COMM_WORLD);
	
	//	Printing the time taken
#ifdef DO_TIMING

	auto finish = chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;

	if (id == 0)
		{
			cout << "The code took " << elapsed.count() << "s to run" << endl;
		
		}

#endif

	// Freeing the different data types created
	MPI_Type_free(&Datatype_left_send);
	MPI_Type_free(&Datatype_right_send);
	MPI_Type_free(&Datatype_top_send);
	MPI_Type_free(&Datatype_bottom_send);
	MPI_Type_free(&Datatype_top_recv);
	MPI_Type_free(&Datatype_bottom_recv);
	MPI_Type_free(&Datatype_left_recv);
	MPI_Type_free(&Datatype_right_recv);

	
	// deleting all the data stored
	delete[] grid_2D;
	delete[] grid_1D;
	delete[] old_grid_2D;
	delete[] old_grid_1D;
	delete[] new_grid_2D;
	delete[] new_grid_1D;

    MPI_Finalize();
}
