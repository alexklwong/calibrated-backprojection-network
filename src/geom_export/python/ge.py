# ge - Package for export of 3D geometry
#
# (c) 2020-11-23 Martin Matousek
# Last change: $Date$
#              $Revision$

import numpy as np

class Ge():
    """ 3D geometry export - base class defining the interface. """

    def close( this ):
        raise Exception( "Unimplemented." )

    def points( this ):
        raise Exception( "Unimplemented." )

    @staticmethod
    def points_arg_helper( X, color=None, colortype=None ):

        npt = X.shape[1]

        if not color is None:
            if isinstance( color, list ) or isinstance( color, tuple ):
                color = np.array( [ color ] ).T

            nc = color.shape[1]

            if( len( color.shape ) != 2 or color.shape[0] != 3 or
                        ( nc != npt and nc != 1 ) ):
                raise Exception( "The color must be 3x1 or 3xn array." )

            if nc == 1:
                color = np.tile( color, ( 1, npt ) )

            if color.dtype != 'float' and color.dtype != 'uint8':
                raise Exception( "Unhandled data type of the color(s)." )

            if colortype is None:
                pass # no conversion, keep as is

            elif colortype == 'uint8':
                if color.dtype == 'float':
                    color = ( color * 255.0 ).astype( 'uint8' )

            elif colortype == 'double':
                if color.dtype == 'uint8':
                    color = ( color * 255.0 ).astype( 'uint8' )
            else:
                raise Exception( "Unhandled value for colortype." )


        bad = np.isnan( X ).any( axis=0)

        if bad.any():
            ok = ~bad
            X = X[:,ok]

            if not color is None:
                color = color[:,ok]

        return X, color

class GePly( Ge ):
    """ 3D geometry export into PLY file. """

    def __init__( this, file, fmt='binary' ):
        """
        Constructor.

        obj = GePly( file, fmt='binary' )

          fmt - 'binary' or 'ascii'
        """

        if fmt != 'binary' and fmt != 'ascii':
            raise Exception( 'Unknown ply format requested.' )

        this.fh = open( file, 'wt' ) # opened ply file handle
        this.binary = fmt == 'binary' # PLY format: true = binary, false = ascii

        this.vertices = [] # list of subarrays of vertices
        this.colors = []   # list of subarrays of corresponding vertex colours
        this.vcount = 0    # total count of vertices

        if this.binary:
            print( 'ply\nformat binary_little_endian 1.0\n',
                    file=this.fh, end='' )
        else:
            print( 'ply\nformat ascii 1.0\n', file=this.fh, end=''  )

    def close( this ):
        """
        Finish and close the PLY file.

        obj.close()
        """

        if this.fh is None:
            return

        is_colors = False
        for c in this.colors:
            if not c is None:
                is_colors = True

        if is_colors:
            for ci in range( len( this.colors ) ):
                if this.colors[ci] is None:
                    c = np.ones( np.shape( this.vertices[xi] ) ) * 255
                    this.colors[ci] = c.astype( 'uint8' )


        # write head (colours are used only if needed)
        print( 'element vertex', this.vcount, file=this.fh )
        print( 'property float x', file=this.fh )
        print( 'property float y', file=this.fh )
        print( 'property float z', file=this.fh )
        if is_colors:
            print( 'property uchar red', file=this.fh )
            print( 'property uchar green', file=this.fh )
            print( 'property uchar blue', file=this.fh )


        print( 'end_header', file=this.fh )

        # write data
        if this.binary:
            # vertices
            for i in range( len( this.vertices ) ):
                v = this.vertices[i].astype( 'float32' ).view( 'uint8' )
                if is_colors:
                    c = this.colors[i].view( 'uint8' )
                    v = np.hstack( ( v, c ) )
                v.tofile( this.fh )

        else:
            # vertices
            for i in range( len( this.vertices ) ):
                v = this.vertices[i]
                if is_colors:
                    c = this.colors[i]
                    v = np.hstack( ( v, c ) )
                    np.savetxt( this.fh, v, '%f %f %f %i %i %i' )
                else:
                    np.savetxt( this.fh, v, '%f %f %f' )


        this.fh.close()
        this.fh = None

    def points( this, X, color=None ):
        """
        Export of points.

        obj.points( X, color=None )

        color: None or 3x1 or 3xN numpy array of rgb values
        """

        X, color = this.points_arg_helper( X, color, colortype='uint8' )

        this.vcount += X.shape[1]

        this.vertices += [ np.ndarray.copy( X.T, order='C' ) ]

        if color is None:
            this.colors += [ color ]
        else:
            this.colors += [ np.ndarray.copy( color.T, order='C' ) ]
