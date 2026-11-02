#!/usr/bin/env bash

# Stop script if an error is encountered
set -e

# Set parsed arguments
PLATO_WORKDIR=$1

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
    echo "NOTE: Only for Linux UNIX users!"
    echo ""    
    exit 1
    
else
    # Sync for platonium
    uv sync --all-groups
    
    # Source the environment
    source .venv/bin/activate
    
    # Set global parameters
    PLATO_SETUP=$VIRTUAL_ENV/.bash_profile
    PLATO_PROJECT_HOME=$PWD
    PLATONIUM=$PLATO_PROJECT_HOME/python/platosim/platonium
    
    # Create setup file
    if [ -f "$PLATO_SETUP" ]; then
	rm $PLATO_SETUP
    fi
    touch $PLATO_SETUP
    echo "#!/usr/bin/env bash"                                      >> $PLATO_SETUP
    echo "export PLATO_PROJECT_HOME=$PWD"                           >> $PLATO_SETUP 
    echo "export PLATO_WORKDIR=$PLATO_WORKDIR"                      >> $PLATO_SETUP
    echo "export PYTHONPATH=$PYTHONPATH:$PLATO_PROJECT_HOME/python" >> $PLATO_SETUP
    echo "source $VIRTUAL_ENV/bin/activate"                         >> $PLATO_SETUP
            
    # Add Write a bash_profile that will be loaded
    echo ""                                          >> $HOME/.bashrc
    echo "# >>> -- Export all PlatoSim paths -- <<<" >> $HOME/.bashrc
    echo "source $PLATO_SETUP"                       >> $HOME/.bashrc
    echo "# >>> ------------------------------- <<<" >> $HOME/.bashrc
    
    # Add scripts to global executables
    cp -rf $PLATONIUM/picsim.py             $VIRTUAL_ENV/bin/picsim
    cp -rf $PLATONIUM/varsim.py             $VIRTUAL_ENV/bin/varsim
    cp -rf $PLATONIUM/payload.py            $VIRTUAL_ENV/bin/payload
    cp -rf $PLATONIUM/platonium.py          $VIRTUAL_ENV/bin/platonium
    cp -rf $PLATONIUM/../scripts/migtool.py $VIRTUAL_ENV/bin/migtool
    cp -rf $PLATONIUM/../scripts/pdshow.py  $VIRTUAL_ENV/bin/pdshow
    cp -rf $PLATONIUM/../scripts/yamlio.py  $VIRTUAL_ENV/bin/yamlio
    cp -rf $PLATONIUM/../scripts/starcat2picsim.py  $VIRTUAL_ENV/bin/starcat2picsim

    chmod +x $VIRTUAL_ENV/bin/picsim
    chmod +x $VIRTUAL_ENV/bin/varsim
    chmod +x $VIRTUAL_ENV/bin/payload
    chmod +x $VIRTUAL_ENV/bin/platonium
    chmod +x $VIRTUAL_ENV/bin/migtool
    chmod +x $VIRTUAL_ENV/bin/pdshow
    chmod +x $VIRTUAL_ENV/bin/yamlio
    chmod +x $VIRTUAL_ENV/bin/starcat2picsim
    
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
    echo ">> argos -h"    
    echo ">> pdshow -h"
    echo ">> yamlio -h"
    echo ">> migtool -h"
    echo ">> starcat2picsim -h"
    echo ""

    # Reload .bashrc
    bash
fi
