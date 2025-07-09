#!/bin/sh

echo "-=--------------------------- env shell script"
#eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
source /home/navidkrad/work/i3kiss/i3setup_extended.sh
echo running command:
echo $@
/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.8.2/env-shell.sh $@
