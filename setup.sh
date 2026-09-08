#!/usr/bin/env bash

# Stop script if an error is encountered
set -e

# Set parsed arguments
PLATO_WORKDIR=$1

# Set global parameters
POETRY=$HOME/.local/bin
PLATO_SETUP=$PWD/.bash_profile
PLATO_PROJECT_HOME=$PWD
PLATONIUM=$PLATO_PROJECT_HOME/python/platosim/platonium

# If no arguments are given write usage message
if [ -z "$PLATO_WORKDIR" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then    
    echo ""
    echo "Usage: setup.sh"
    echo ""
    echo "ARGUMENTS:"
    echo "</path/to/plato_workdir>  : Location to PlatoSim working directory"
    echo ""
    echo "DESCRIPTION:"
    echo "PLATOnium is a PlatoSim toolkit that allows the user to run multi-camera"
    echo "simulations and while also using the LESIA pipeline in extention to extract"
    echo "on-ground or on-board data products in accordance to the mission strategy."
    echo "Given the PLATO_WORDIR as input arguments, this script sets up your path"
    echo "environments so they are globally defined on your system."
    echo ""
    echo "NOTE:"
    echo "Only for Linux UNIX users!"
    echo ""    
    exit 1
    
else

    # SETUP PLATONIUM
    
    # Footprint of project setup (overwrite)
    if [ -f "$PLATO_SETUP" ]; then
	rm $PLATO_SETUP
    fi
    
    # Create setup file
    touch $PLATO_SETUP

    # Export paths
    echo "#!/usr/bin/env bash"                                      >> $PLATO_SETUP
    echo "export POETRY=$POETRY"                                    >> $PLATO_SETUP
    echo "export PLATO_PROJECT_HOME=$PWD"                           >> $PLATO_SETUP 
    echo "export PLATO_WORKDIR=$PLATO_WORKDIR"                      >> $PLATO_SETUP
    echo "export PYTHONPATH=$PYTHONPATH:$PLATO_PROJECT_HOME/python" >> $PLATO_SETUP
    
    # Add Write a bash_profile that will be loaded
    #if grep -q "source $PLATO_SETUP" "$HOME/.bashrc"; then
    echo ""                                          >> $HOME/.bashrc
    echo "# >>> -- Export all PlatoSim paths -- <<<" >> $HOME/.bashrc
    echo "source $PLATO_SETUP"                       >> $HOME/.bashrc
    echo "# >>> ------------------------------- <<<" >> $HOME/.bashrc

    # Add scripts to global executables
    cp -rf $PLATONIUM/picsim.py             $CONDA_PREFIX/bin/picsim
    cp -rf $PLATONIUM/varsim.py             $CONDA_PREFIX/bin/varsim
    cp -rf $PLATONIUM/payload.py            $CONDA_PREFIX/bin/payload
    cp -rf $PLATONIUM/platonium.py          $CONDA_PREFIX/bin/platonium
    cp -rf $PLATONIUM/../scripts/migtool.py $CONDA_PREFIX/bin/migtool
    cp -rf $PLATONIUM/../scripts/pdshow.py  $CONDA_PREFIX/bin/pdshow

    chmod +x $CONDA_PREFIX/bin/picsim
    chmod +x $CONDA_PREFIX/bin/varsim
    chmod +x $CONDA_PREFIX/bin/payload
    chmod +x $CONDA_PREFIX/bin/platonium
    chmod +x $CONDA_PREFIX/bin/migtool
    chmod +x $CONDA_PREFIX/bin/pdshow
    
    # PROLOGUE
        
    # Finish with prolog message
    echo "----------------------------"
    echo " PLATOnium has been set up !"
    echo "----------------------------"
    echo ""
    echo " From bash checkout:"
    echo ""
    echo ">> picsim -h"
    echo ">> varsim -h"
    echo ">> payload -h"
    echo ">> platonium -h"
    echo ""
    echo " See also help functions:"
    echo ""
    echo ">> pdshow -h"    
    echo ">> migtool -h"
    echo ""

    # Reload .bashrc
    bash
fi
