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




int main()
{
    try
    {
        GuiManager app;
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "An unhandled exception occurred: " << e.what() << '\n';
#ifdef _WIN32
        system("pause");
#endif
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "An unknown fatal error occurred." << '\n';
#ifdef _WIN32
        system("pause");
#endif
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
