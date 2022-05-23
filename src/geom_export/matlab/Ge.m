classdef Ge < handle
%GE  3D geometry export - abstract class defining the interface.

% (c) 2017-04-07 Martin Matousek, Czech Technical University in Prague
% Last change: $Date$
%              $Revision$

methods( Abstract )

%GE/CLOSE  Finish and close geometry export target.
%
%   ge.close()
close( this )

%GE/POINTS  Export of points.
%
%   inx = ge.points( X, ... )
%
%   Input:
%     X .. 3xN matrix of 3D points as column vectors
%
%     optional key-value pairs:
%       'color' .. single RGB color vector (same color for all points) or
%                  3xN matrix of RGB colors for every point
%
%   Output:
%     inx .. internal point indices corresponding to X (continuous range)
inx = points( this, X, varargin )

end % meths.

methods( Access = protected )

function [opt, X] = points_arg_helper( ~, X, varargin )
%GE/POINTS_ARG_HELPER  Parsing and check of input arguments for GE/POINTS.
%
%   opt = ge.points_arg_helper( X, ... )
%
%   To be used by derived classes.
%
%   Color arguments are converted to uint8.

opt = parseargs( 'color', [], 'colortype', 'none', varargin{:} );

npt = size( X, 2 );

if( ~isempty( opt.color ) )
  if( isequal( size( opt.color ), [1 3] ) ), opt.color = opt.color'; end

  nc = size( opt.color, 2 );

  if( ( nc ~= npt && nc ~= 1 ) || size( opt.color, 1 ) ~= 3 )
    error( 'color must be 3x1 or 3xn vector' );
  end

  if( size( opt.color, 2 ) == 1 )
    opt.color = opt.color( :, ones( 1, npt ) );
  end

  switch( opt.colortype )
    case 'uint8'
      if( ~isa( opt.color, 'uint8' ) )
        opt.color = uint8( opt.color * 255 );
      end
    case 'double'
      if( isa( opt.color, 'uint8' ) )
        opt.color = double( opt.color ) / 255.0;
      end
    case 'none'
      % keep
    otherwise
      error( 'wrong color type %s', opt.colortype )
  end
end

bad = isnan( X );
if( any( bad(:) ) )
  ok = ~any( bad );
  X = X(:, ok );
  if( ~isempty( opt.color ) )
    opt.color = opt.color( :, ok );
  end
end

end

end % meths.

end % cls.
