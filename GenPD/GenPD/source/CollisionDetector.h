#pragma once

#include "math_headers.h"
#include "opengl_headers.h"

#include <vector>
#include <list>
#include <queue>
#include <algorithm>
#include <cassert>


using namespace std;



/*

	// This struct is for Unit test //

struct MockingTet {
	int id1;
	int id2;
	int id3;
	int id4;
};


struct MockingMesh{
	//tet info
	vector<MockingTet> tet;
	vector<glm::vec3> vertices;
};

*/



// Hash Table element
struct MappedVertice {

	int obj_index;
	int vectice_index;
	int timeStamp;
};

struct MappedFace {

	int obj_index;
	int face_index;
	int timeStamp;
};


//forward declaration
class TetMesh;
class ClothMesh;

//contain Collision information
struct CollisionInfo {

	TetMesh* penetratedMesh = nullptr;
	TetMesh* penetratingMesh = nullptr;

	int penetratedIndex;
	int penetratingIndex;


	//index list if penetrating mesh
	vector<int> verticeList;
	vector<float> penetrationDepth;
	vector<glm::vec3> penetrationDirection;
};


class CollisionDetector
{

	/*
	* collision detection class using spatial hashing
	*/

public:

	// for valid current point
	int timeStamp;


	///// Hash parameter ///////
	int p1, p2, p3;
	int n;//table size
	float gridSize;
	///////////////////////////

	vector<CollisionInfo> previous;


	float epsilone;
	//initialize Large prime number p1,p2,p3, table size: n, gridsize, timestamp, epsilone for floating number comparison
	CollisionDetector() :p1(73856093), p2(19349663), p3(83492791), n(30000), gridSize(1), timeStamp(0), epsilone(0.0000001) {

		// initialize hashTable
		for (int i = 0; i < n; i++) hashTableV.push_back(list<MappedVertice>());
		for (int i = 0; i < n; i++) hashTableF.push_back(list<MappedFace>());



	};
	~CollisionDetector() {
	};


	//add object
	void addObject(TetMesh* mesh);

	//set Hashfunction Parameter
	void setHashParam(int p1, int p2, int p3, int tableSize, float gridSize);
	void setGridSize(float gridSize);

	vector<CollisionInfo> detectCollision(VectorX& x);
	vector<CollisionInfo> detectSelfCollision(VectorX& x);

	//meshList
	vector<TetMesh*> m_obj_list;

	//visualize collision detection visualization
	void DrawCollisionPoint(const VBO& vbos);
	void DrawPenetration(const VBO& vbos);





private:
	enum IsProcessed {
		NON_COLLIDE,
		COLLIDE_NOT_PROCESSED,
		COLLIDE_PROCESSED
	};


	struct PointState {
		float penetrationDepth;
		glm::vec3 penetrationDirection;
		IsProcessed isProcessed;
	};


	//hashtable
	vector<list<MappedVertice>> hashTableV;
	vector<list<MappedFace>> hashTableF;
	vector<vector<vector<int>>> topologyList;
	vector<vector<PointState>> pointStateTable;
	//vertex topology (adjacent vertex)


	void mapVertices(); //map all vertices into the HashTable
	void mapFaces(); //map all faces into the HashTable

	int calculateKey(float x, float y, float z); // calculate hash key


	//geometry computation////

	bool IsIntersectTetPoint(TetMesh* mesh, unsigned int tet_index, glm::vec3& point); // check tetrahedron and point intersection
	bool IsIntersectTriLine(TetMesh* mesh, int tri_index, glm::vec3 pStart, glm::vec3 pEnd, glm::vec3& exactPoint);
	//TODO: isIntersect btwn triangle and line. return should contain exact point and normal

	void calculateTetAABB(TetMesh* mesh, unsigned int tet_index, glm::vec3& minout, glm::vec3& maxout);// calculate aabb of tetrahedron
	void calculateTriAABB(TetMesh* mesh, unsigned int tri_index, glm::vec3& minout, glm::vec3& maxout);
	bool checkSamePoint(glm::vec3& point1, glm::vec3& point2); // check if two point are the same


	void voxelTraversal(glm::vec3 pStart, glm::vec3 pEnd, vector<int>& hashkeylist);


	void cleanHashTable(); // periodically clean hash table

	void makeVectorUnique(vector<int>& v); // erase duplicated element
	void initPointStateTable();
	void computePenetration(vector<CollisionInfo>& collisionInfo);
	void computePenetrationBf(vector<CollisionInfo>& collisionInfo);

	void propagatePenetration();
	float weightAverageD(vector<float>& w, vector<glm::vec3>xi_p, vector<glm::vec3>& n);
	glm::vec3 weightAverageN(vector<float>& w, vector<glm::vec3>& n);
	void processCollidingPoint(int meshIndex, int collidingIndex);
	bool findNormalAndPoint(glm::vec3 startP, glm::vec3 endP, vector<int>& candidateKey, glm::vec3& normal, glm::vec3& exactP);
	void updateCollisionInfo(vector<CollisionInfo>& collisionInfo);
	void findShortest(TetMesh* mesh, glm::vec3 collidingPoint, glm::vec3& shortest);





	int debug_checkReallynoIntersection(glm::vec3 startP, glm::vec3 endP);
	//void initBorderPoint();







};

