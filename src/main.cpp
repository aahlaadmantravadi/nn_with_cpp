// =============================================================================
// File: src/main.cpp
// =============================================================================
//
// Description: The main entry point for the "TensorFlow from Scratch"
//              application. Its sole responsibility is to create and run the
//              main GUI manager, which handles the entire application lifecycle.
//
// =============================================================================

#include "gui/GuiManager.h"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        // GuiManager encapsulates the entire application logic.
        // Creating an instance and running it is all that's needed.
        GuiManager app;
        app.run();
    } catch (const std::exception& e) {
        // Catch any standard exceptions that might occur during initialization
        // or runtime and report them.
        std::cerr << "An unhandled exception occurred: " << e.what() << std::endl;
        // Pause to allow user to see the error message in the console.
        #ifdef _WIN32
            system("pause");
        #endif
        return EXIT_FAILURE;
    } catch (...) {
        // Catch any other unknown exceptions.
        std::cerr << "An unknown fatal error occurred." << std::endl;
        #ifdef _WIN32
            system("pause");
        #endif
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
