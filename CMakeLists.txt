cmake_minimum_required(VERSION 3.23)
project(yaalp)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/render)
include_directories(include/simulation)

add_executable(yaalp
        include/render/Viewpart.h
        include/render/WorldRenderer.h
        include/simulation/Entity.h
        include/simulation/Food.h
        include/simulation/GameManager.h
        include/simulation/Individu.h
        include/simulation/IndividuSprite.h
        include/simulation/IndivSprite.h
        include/simulation/NeuralNetwork.h
        include/simulation/obstacles.h
        include/simulation/ObstacleSprite.h
        include/simulation/PlantSprite.h
        include/simulation/Sprite.h
        include/simulation/World.h
        src/render/Viewpart.cpp
        src/render/WorldRenderer.cpp
        src/simulation/Entity.cpp
        src/simulation/Food.cpp
        src/simulation/GameManager.cpp
        src/simulation/Individu.cpp
        src/simulation/IndividuSprite.cpp
        src/simulation/IndivSprite.cpp
        src/simulation/NeuralNetwork.cpp
        src/simulation/obstacles.cpp
        src/simulation/ObstacleSprite.cpp
        src/simulation/PlantSprite.cpp
        src/simulation/Sprite.cpp
        src/simulation/World.cpp)
