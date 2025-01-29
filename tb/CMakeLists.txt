# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(tb)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})


include_directories(/usr/local/include)
include_directories(/usr/local/include/cryptopp)

set(SOURCES
    ende.cpp
    ModelCrypto.cpp
)

set(HEADERS
    ModelCrypto.h
)

qt4_wrap_cpp(SOURCES ${HEADERS})


add_executable(${PROJECT_NAME}_ende
    ${SOURCES})

target_link_libraries(${PROJECT_NAME}_ende ${QT_LIBRARIES} /usr/local/lib/libcryptopp.a)



set(SOURCES1
    main.cpp
    lcTensorBoard.cpp
    ViewTensorBoard.cpp
    ViewTimeSeries.cpp
    qcustomplot.cpp
    PropsSection.cpp
)

set(HEADERS1
    lcTensorBoard.h
    ViewTensorBoard.h
    ViewTimeSeries.h
    qcustomplot.h
    PropsSection.h
)

qt4_wrap_cpp(SOURCES1 ${HEADERS1})


add_executable(${PROJECT_NAME}
    ${SOURCES1})

target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES} /usr/local/lib/libcryptopp.a)


